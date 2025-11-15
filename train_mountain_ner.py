import argparse
import os
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

# Label scheme used by prepare_mountain_ner.py
LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_or_split_dataset(hf_dataset_dir: str, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42):
    """
    Load dataset saved to disk.
    """
    ds = load_from_disk(hf_dataset_dir)

    # If it is already a DatasetDict with splits, return it
    if isinstance(ds, DatasetDict):
        return ds

    # If a single Dataset, create splits
    if isinstance(ds, Dataset):
        total_test = test_size
        total_val = val_size
        if (total_test + total_val) >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
        
        # Split out test
        split1 = ds.train_test_split(test_size=total_test, seed=seed)
        train_plus = split1["train"]
        test_ds = split1["test"]

        # Compute fraction for validation relative to remaining training set
        val_fraction_of_train_plus = total_val / (1.0 - total_test) if (1.0 - total_test) > 0 else 0.0
        if val_fraction_of_train_plus > 0:
            split2 = train_plus.train_test_split(test_size=val_fraction_of_train_plus, seed=seed)
            train_ds = split2["train"]
            val_ds = split2["test"]
        else:
            train_ds = train_plus
            val_ds = Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})  # empty
        return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    raise ValueError("Loaded object is not a Dataset or DatasetDict. Path: " + str(hf_dataset_dir))


def align_predictions_and_labels(predictions, label_ids):
    """
    predictions: np.array (batch, seq_len, num_labels) or (seq_len, num_labels)
    label_ids: np.array (batch, seq_len) with -100 for tokens to ignore
    Returns lists of predicted label strings and true label strings (for seqeval/evaluate)
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_list, labels_list = [], []
    for i in range(batch_size):
        pred_labels, true_labels = [], []
        for j in range(seq_len):
            if label_ids[i, j] == -100:
                continue
            pred_labels.append(ID2LABEL[int(preds[i, j])])
            true_labels.append(ID2LABEL[int(label_ids[i, j])])
        preds_list.append(pred_labels)
        labels_list.append(true_labels)
    return preds_list, labels_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_dir", required=True, help="Directory of HF dataset saved via Dataset.save_to_disk")
    parser.add_argument("--model_name_or_path", default="bert-base-cased")
    parser.add_argument("--output_dir", default="mountain-ner-output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if available")
    args = parser.parse_args()

    # Load dataset saved to disk
    dataset_dict = load_or_split_dataset(args.hf_dataset_dir, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    print("Dataset splits:", list(dataset_dict.keys()))
    print("Sizes:", {k: len(dataset_dict[k]) for k in dataset_dict.keys()})

    # Check that expected fields exist
    for split in dataset_dict:
        sample = dataset_dict[split][0]
        if "input_ids" not in sample or "attention_mask" not in sample or "labels" not in sample:
            raise ValueError(f"Dataset split '{split}' does not contain required fields 'input_ids','attention_mask','labels'. Inspect the saved dataset.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(LABEL_LIST),
        id2label={str(i): l for i, l in ID2LABEL.items()},
        label2id={l: i for i, l in enumerate(LABEL_LIST)}
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        fp16=args.fp16,
    )

    # evaluation metric via evaluate/seqeval
    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        if isinstance(logits, tuple):  # HF may return tuple
            logits = logits[0]
        preds_list, labels_list = align_predictions_and_labels(np.array(logits), np.array(label_ids))
        results = metric.compute(predictions=preds_list, references=labels_list)
        
        # metric.compute returns nested dict; extract sensible scalars
        overall_precision = results.get("overall_precision", 0.0)
        overall_recall = results.get("overall_recall", 0.0)
        overall_f1 = results.get("overall_f1", 0.0)
        return {"precision": overall_precision, "recall": overall_recall, "f1": overall_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()