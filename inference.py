import gradio as gr
from transformers import pipeline

MODEL_DIR = "output_mountain_ner"
nlp = pipeline(
    "ner",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple"
)

def extract_entities(text):
    entities = nlp(text)
    spans = []
    last_idx = 0
    
    for e in entities:
        start, end, label = e["start"], e["end"], e["entity_group"]
        if start > last_idx:
            spans.append((text[last_idx:start], None))
        spans.append((text[start:end], label))
        last_idx = end

    if last_idx < len(text):
        spans.append((text[last_idx:], None))

    return spans

description = """
Enter some text about mountains\n

```
Examples:
Mount Everest towers majestically over the Himalayas, attracting climbers from all over the world.
The view from Mount Kilimanjaro at sunrise is absolutely breathtaking.
Hikers often visit Mount Fuji to catch the cherry blossoms in full bloom around its base.
Mount Elbrus, the highest peak in Europe, challenges even the most experienced climbers.
The Andes stretch along South America, with Aconcagua standing as their tallest peak.
The Alps are dotted with charming villages, making Mont Blanc a popular destination for tourists.
Denali, formerly known as Mount McKinley, rises dramatically above Alaska’s wilderness.
The Carpathian Mountains are famous for their dense forests and wildlife, including bears and wolves.
Mount Rainier dominates the skyline in Washington State, with glaciers cascading down its slopes.
```
"""

iface = gr.Interface(
    fn=extract_entities,
    inputs=gr.Textbox(label="Input text"),
    outputs=gr.HighlightedText(),
    title="Mountain NER",
    description=description
)

iface.launch()