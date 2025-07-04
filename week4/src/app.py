# save this as caption_demo.py

import gradio as gr
import requests
from PIL import Image
from io import BytesIO

# === Stub: pretend we have 5 models ===
MODEL_NAMES = [
    "Model A (Transformer)",
    "Model B (ViT + LSTM)",
    "Model C (CLIP)",
    "Model D (BLIP)",
    "Model E (Custom CNN-RNN)"
]

def get_random_image():
    url = "https://picsum.photos/800/600"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def generate_captions(image):
    results = []
    for name in MODEL_NAMES:
        # Replace this stub with your actual inference later
        caption = f"This is a dummy caption by {name}."
        results.append((name, caption))
    return results

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
                row_count=(5, "fixed"),
                interactive=False,
                wrap=True,
                label="Captions"
            )

    get_random_btn.click(fn=get_random_image, outputs=image_input)
    generate_btn.click(fn=generate_captions, inputs=image_input, outputs=output_table)

demo.launch(server_name="0.0.0.0", server_port=7860)
