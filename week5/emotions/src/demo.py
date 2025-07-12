import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
import gradio as gr
import random
import pandas as pd
import numpy as np
import subprocess

from model import MLPClassifier, TinyTransformerClassifier


VIDEO_MAP = {
    "Emotional damage": "videos/emotional_damage.mp4",
    "Chocolate": "videos/chocolate.mp4",
    "It do go down": "videos/it_do_go_down.mp4",
    "Surprise surprise": "videos/surprise_surprise.mp4",
    "Happy happy happy": "videos/happy_happy_happy.mp4",
    "Bezos laugh": "videos/bezos_laugh.mp4",
    "Inflation hit": "videos/inflation_hit.mp4",
    "Nice": "videos/nice.mp4",
}

emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}


DEVICE = torch.device("mps")

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
whisper_model = whisper_model.eval().to(DEVICE)

cls_model = MLPClassifier(768, len(emotion_map))  # TODO don't hardcode these parameters
cls_model.load_state_dict(torch.load("mlp.pt", map_location="cpu"))
cls_model = cls_model.eval().to(DEVICE)


def load_audio_from_mp4(mp4_file, target_sr=16000):
    command = [
        "ffmpeg",
        "-i", mp4_file,
        "-f", "f32le",    # raw 32-bit float PCM
        "-acodec", "pcm_f32le",
        "-ac", "1",       # mono
        "-ar", str(target_sr),
        "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_audio = process.stdout.read()  # read all bytes
    audio_tensor = torch.frombuffer(raw_audio, dtype=torch.float32)
    return audio_tensor, target_sr


def predict_emotions(video_choice):
    waveform, sr = load_audio_from_mp4(VIDEO_MAP[video_choice], target_sr=16000)

    # # TODO Avoid code duplication
    # if waveform.shape[0] > 1:
    #     waveform = torch.mean(waveform, dim=0, keepdim=True)
    # waveform = waveform.squeeze().numpy()

    # if sr != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    #     waveform = resampler(torch.tensor(waveform)).squeeze().numpy()
    #     sr = 16000

    inputs = whisper_processor(waveform.numpy(), sampling_rate=sr, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        encoder_out = whisper_model.encoder(**inputs)
    states = encoder_out.last_hidden_state[0].cpu().numpy()  # [seq_len, hidden_dim]
    X_pooled = np.mean(states, axis=0, keepdims=True)
    X_pooled = torch.tensor(X_pooled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        y_pred = cls_model(X_pooled)
    probs = torch.softmax(y_pred, dim=1).cpu().numpy().flatten()

    percents = [round(p * 100, 1) for p in probs]

    df = pd.DataFrame({
        "Emotion": [emotion_map[i] for i in range(1, 9)],
        "Probability": percents,
        "Label": [f"{p}%" for p in percents],
    })

    return df


def switch_video(video_choice):
    return gr.update(value=VIDEO_MAP[video_choice]), gr.update(value=None)


with gr.Blocks(css="""
body {
    background: radial-gradient(circle at center, #ff00cc, #3333ff);
    color: #fff;
    font-family: Comic Sans MS, cursive;
}
h1 {
    font-size: 2.2em;
    text-shadow: 1px 1px #000;
    text-align: center;
}
""") as demo:
    gr.Markdown("# 🤯 EMOTIONAL DAMAGE 🤯")

    video_choice = gr.Dropdown(
        choices=list(VIDEO_MAP.keys()),
        label="Select Your POISON",
        value="Emotional damage"
    )

    video_player = gr.Video(
        value=VIDEO_MAP["Emotional damage"],
        interactive=False,
        height=360,
        width=640
    )

    detect_btn = gr.Button("☢️ UNLEASH MAXIMUM DAMAGE ☢️")

    emotion_plot = gr.BarPlot(
        value=None,
        x="Probability",
        y="Emotion",
        color="Probability",
        orientation="horizontal",
        title="Emotion Probabilities",
        x_title="%",
        y_title="Emotion",
        color_scale=["#ff00ff", "#00ffff", "#ffff00"],
        tooltip=["Emotion", "Probability", "Label"],
        width=640,
        height=400
    )

    video_choice.change(
        switch_video,
        inputs=video_choice,
        outputs=[video_player, emotion_plot]
    )

    detect_btn.click(
        predict_emotions,
        inputs=video_choice,
        outputs=[emotion_plot]
    )

# demo.launch(share=True)
demo.launch()
