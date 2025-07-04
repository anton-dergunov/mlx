# caption_demo_infer.py

import gradio as gr
import requests
import yaml
import torch
from PIL import Image
from io import BytesIO

from transformers import CLIPProcessor, CLIPModel
from model import ImageCaptioningModel
from utils import get_device

# --------- LOAD CONFIG ---------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

device = get_device()
print(f"Using device: {device}")

# --------- LOAD CLIP ---------
CLIP_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME, use_fast=False)
clip_model.eval()

# --------- LOAD DECODER MODELS ---------
decoder_models = []
model_names = []

for model_cfg in cfg["models"]:
    name = model_cfg["name"]
    decoder_type = model_cfg["decoder"]
    path = model_cfg["path"]

    print(f"Loading model: {name} ({decoder_type}) from {path}")

    model = ImageCaptioningModel(decoder_type=decoder_type)
    checkpoint = torch.load(path, map_location="cpu")  # avoid error while loading with MPS

    # Fix for my refactoring. Remap keys: replace "gpt2_model." -> "lm."
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace("gpt2_model.", "lm.")
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    decoder_models.append(model)
    model_names.append(name)

print(f"Loaded {len(decoder_models)} decoder models.")


# --------- RANDOM IMAGE HELPER ---------
def get_random_image():
    url = "https://picsum.photos/800/600"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


# --------- INFERENCE LOGIC ---------
def generate_captions(image):
    if image is None:
        return []

    # === 1) Get CLIP embedding ===
    processed = clip_processor(images=image, return_tensors="pt")
    processed = {k: v.to(device) for k, v in processed.items()}

    with torch.no_grad():
        image_embed = clip_model.get_image_features(**processed)

    # === 2) Run each model ===
    results = []
    for model, name in zip(decoder_models, model_names):
        with torch.no_grad():
            generated_seqs = model.generate(image_embed)
            generated_text = model.tokenizer.decode(generated_seqs[0], skip_special_tokens=True)
        results.append((name, generated_text))

    return results


# --------- GRADIO UI ---------
with gr.Blocks(css="footer {display: none !important;}") as demo:
    gr.Markdown("# 🏷️ Image Captioning Demo\nUpload your own image, or get a random one!")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            get_random_btn = gr.Button("🔁 Get Random Image")
            generate_btn = gr.Button("▶️ Generate Captions")

        with gr.Column():
            output_table = gr.Dataframe(
                headers=["Model", "Caption"],
                datatype=["str", "str"],
                row_count=(len(model_names), "fixed"),
                interactive=False,
                wrap=True,
                label="Captions"
            )

    get_random_btn.click(fn=get_random_image, outputs=image_input)
    generate_btn.click(fn=generate_captions, inputs=image_input, outputs=output_table)

demo.launch(server_name="0.0.0.0", server_port=7860)
