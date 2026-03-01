
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_ID = "google/medgemma-1.5-4b-it"
processor, model = None, None

def load_model():
    global processor, model
    print("Loading MedGemma 1.5 4B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Model ready ✔")

def run_inference(prompt: str, image=None):
    content = []
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True)
