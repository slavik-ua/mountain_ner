# Mountain NER Model

This project trains a **Named Entity Recognition (NER)** model specifically for recognizing **mountain names** in text.
It is built for **learning and experimentation**, using a custom dataset you can recreate.

This project includes:
- Dataset generation notebook
- Training script
- Inference example

---

## 📘 Dataset
The creation process is documented in:
- `dataset_generation.ipynb` explains the **dataset creation process**, including tokenization, label assignment, and preprocessing.  

The unfinished dataset is available at https://huggingface.co/datasets/slavik-ua/mountain_ner

> **Note:** This dataset is **preliminary** and intended for **learning and experimentation**.

---

## 🔧 Setup & Requirements

Install the required Python packages:

```sh
pip install -r requirements.txt
```

This notebook is configured to query the local model for text processing. Ensure the server is running or uncomment

---

# 📚 Dataset Generation
Run the Jupyter notebook to generate or preprocess the dataset:
```sh
jupyter notebook dataset_generation.ipynb
```
The dataset generation notebook is **already configured to use a local GGUF model** running on **llama-server**.

---

Using the local model
1. Download your GGUF model
2. Run the model server:
```sh
llama-server -m model.gguf
```
3. The notebook will automatically connect to the local server.

---

Using a hosted model
If you prefer to use a hostem LLM instead of the local model
1. Uncomment the following lines in the notebook:
```python
#from langchain_google_genai import ChatGoogleGenerativeAI
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
```
2. Make sure your .env file contains the necessary varible
```sh
GOOGLE_API_KEY=api_key
```
3. Load you API key:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

# 🏋️ Training the model

Train the NER model using the provided script:
```sh
python train_mountain_ner.py \
    --hf_dataset_dir mountains_prepared.hf_dataset/ \
    --model_name_or_path bert-base-cased \
    --output_dir output_mountain_ner \
    --num_train_epochs 4
```

This will create a directory **output_mountain_ner/** containing:
- model weights
- tokenizer
- config
- training logs

---

# 🚀 Inference example
You can run inference on the trained Mountain NER model in 2 ways:
1. Using the Hugging Face pipeline
```python
from transformers import pipeline

MODEL_DIR = "output_mountain_ner"
nlp = pipeline(
    "ner",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple"
)

sentence = "Mount Everest towers majestically over the Himalayas, attracting climbers from all over the world."
print(nlp(sentence))
```
2. Using the example Gradio script
There is an unfinished example Grdio script called **inference.py**.
**Note:** this script is unfinished and may require some modifications.


---

# 📚 Wikipedia Attribution (Required by CC BY-SA 4.0)
This project uses mountain names extracted from the Wikipedia page
“List of mountains by elevation” (CC BY-SA 4.0).
© Wikipedia contributors. Content is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License

---

# License

- Code: MIT License  
- Dataset (Wikipedia-derived): CC BY-SA 4.0  
- Pretrained GGUF model: follow original license

---


# 📝 Notes
- The dataset was initially generated using Mistal-Nemo-Instruct-2407.Q4_K_S: https://huggingface.co/QuantFactory/Mistral-Nemo-Instruct-2407-GGUF/tree/main
- This project and **incomplete** dataset are intended for **learning and experimentation**.