import gradio as gr
import requests
import yaml
import torch
from PIL import Image
from io import BytesIO
import argparse
from functools import partial

from transformers import CLIPProcessor, CLIPModel
from model import ImageCaptioningModel
from utils import get_device

CLIP_NAME = "openai/clip-vit-base-patch32"

# --------- LOAD CLIP ---------
def load_clip(device):
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME, use_fast=False)
    clip_model.eval()
    return clip_model, clip_processor

# --------- LOAD DECODER MODELS ---------
def load_models(cfg, device):
    decoder_models = []
    model_names = []

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        decoder_type = model_cfg["decoder"]
        path = model_cfg["path"]

        print(f"Loading model: {name} ({decoder_type}) from {path}")

        model = ImageCaptioningModel(decoder_type=decoder_type)
        checkpoint = torch.load(path, map_location="cpu")

        # Fix: remap keys if refactored from 'gpt2_model.' to 'lm.'
        new_state_dict = {k.replace("gpt2_model.", "lm."): v for k, v in checkpoint.items()}

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        decoder_models.append(model)
        model_names.append(name)

    print(f"Loaded {len(decoder_models)} decoder models.")
    return decoder_models, model_names

# --------- RANDOM IMAGE HELPER ---------
def get_random_image():
    url = "https://picsum.photos/800/600"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# --------- INFERENCE LOGIC WITH STREAMING ---------
def generate_captions_stream(clip_processor, clip_model, decoder_models, model_names, device, image):
    if image is None:
        yield "❌ No image provided. Please upload or get a random image."
        return

    full_md = "## 📝 Caption Progress\n\n"

    full_md += "🔄 **Step 1:** Encoding image with CLIP...\n\n"
    yield full_md

    processed = clip_processor(images=image, return_tensors="pt")
    processed = {k: v.to(device) for k, v in processed.items()}

    with torch.no_grad():
        image_embed = clip_model.get_image_features(**processed)

    full_md += "✅ **Step 2:** Got CLIP embedding. Running caption models...\n\n"

    # Table header
    full_md += "| 🧩 Model | 📝 Caption |\n|---|---|\n"

    yield full_md  # yield header + step info

    results = []
    for idx, (model, name) in enumerate(zip(decoder_models, model_names)):
        # Status for this model
        status_line = f"⏳ **[{idx + 1}/{len(model_names)}]** Running **{name}**...\n\n"

        yield full_md + status_line  # show table so far + status

        with torch.no_grad():
            generated_seqs = model.generate(image_embed)
            generated_text = model.tokenizer.decode(generated_seqs[0], skip_special_tokens=True)

        results.append((name, generated_text))

        # Add this row to the table
        full_md += f"| `{name}` | {generated_text} |\n"

        remaining = len(model_names) - len(results)
        if remaining > 0:
            progress_note = f"\n🔄 Still computing {remaining} more model(s)..."
        else:
            progress_note = "\n✅ **All models done!**"

        yield full_md + progress_note
        
# --------- GRADIO UI ---------
def gradio_ui(generate_captions_fn, model_names, share):
    with gr.Blocks(css="""
        footer {display: none !important;}
        .gradio-container { font-family: 'Segoe UI', sans-serif; }
        h1 { font-size: 1.6em; margin-bottom: 0.5em; }
    """) as demo:
        gr.Markdown("# 🏷️ **Image Captioning Demo**\nUpload an image or grab a random one. Then click ▶️ to see different models generate captions in real-time!")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                get_random_btn = gr.Button("🔁 Get Random Image")
                generate_btn = gr.Button("▶️ Generate Captions")

            with gr.Column():
                output_md = gr.Markdown(label="Captions Progress")

        # Clear output on random or upload
        get_random_btn.click(
            fn=lambda: (get_random_image(), "🔄 Ready for inference..."),
            outputs=[image_input, output_md]
        )

        image_input.change(
            fn=lambda x: "🗑️ Cleared previous captions. Ready for new image.",
            inputs=image_input,
            outputs=output_md
        )

        generate_btn.click(
            fn=generate_captions_fn,
            inputs=image_input,
            outputs=output_md
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)

# --------- MAIN ENTRY ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--public-link",
        action="store_true",
        default=False,
        help="Create a public link to make app accessible externally"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    clip_model, clip_processor = load_clip(device)
    decoder_models, model_names = load_models(cfg, device)

    generate_captions_fn = partial(
        generate_captions_stream,
        clip_processor, clip_model, decoder_models, model_names, device
    )

    gradio_ui(generate_captions_fn, model_names, args.public_link)

if __name__ == "__main__":
    main()
