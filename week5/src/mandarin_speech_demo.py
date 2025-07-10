# pip install gradio whisper praat-parselmouth numpy matplotlib pypinyin torch transformers
#
# for MFA:
# pip install spacy-pkuseg dragonmapper hanziconv
# mfa model download acoustic mandarin_mfa
# mfa model download dictionary mandarin_mfa
# not needed: mfa model download language_model mandarin_mfa_lm


import gradio as gr
import torch
from transformers import pipeline
import tempfile
import os
import subprocess
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from pypinyin import pinyin, Style
import shutil


# 1️⃣ Load Whisper
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
pipeline = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=torch.float16,
    device=DEVICE
)

# 2️⃣ Target phrase
TARGET_TEXT = "我喜欢机器学习"
TARGET_PINYIN = ' '.join(sum(pinyin(TARGET_TEXT, style=Style.TONE3), []))

# 3️⃣ MFA setup
MFA_MODEL = "mandarin_mfa"  # download from MFA site
MFA_DICT = os.path.expanduser("/Users/anton/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict")

# TTS: optional, use Coqui or Bark here — or skip for now
def synthesize():
    return "demo_tts_output.wav"

# 4️⃣ Recognize & analyze
def analyze(audio):
    audio_path = audio  # already a file path

    # Whisper
    result = pipeline(audio_path)
    hanzi = result["text"]
    pinyin_out = ' '.join(sum(pinyin(hanzi, style=Style.TONE3), []))

    # ---- Prepare CORPUS_DIRECTORY ----
    corpus_dir = tempfile.mkdtemp()
    audio_basename = "utt1.wav"
    corpus_audio_path = os.path.join(corpus_dir, audio_basename)
    shutil.copy(audio_path, corpus_audio_path)

    text_file_path = os.path.join(corpus_dir, "utt1.lab")
    with open(text_file_path, "w") as f:
        f.write(TARGET_TEXT)

    # ---- Run MFA align ----
    mfa_output_dir = tempfile.mkdtemp()
    print(mfa_output_dir)
    subprocess.run([
        "mfa", "align",
        corpus_dir,
        MFA_DICT,
        MFA_MODEL,
        mfa_output_dir
    ])

    # ---- Parselmouth pitch ----
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_times = pitch.xs()

    fig, ax = plt.subplots()
    ax.plot(pitch_times, pitch_values)
    ax.set_title("Pitch Contour")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(plot_path.name)

    return f"### Whisper Hanzi\n{hanzi}\n\n### Pinyin\n{pinyin_out}\n\n### Target Pinyin\n{TARGET_PINYIN}", plot_path.name
# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Mandarin Pronunciation Demo")
    gr.Markdown(f"Target phrase: **{TARGET_TEXT}**\n\nExpected Pinyin: `{TARGET_PINYIN}`")

    with gr.Row():
        tts_btn = gr.Button("Play Example Audio")
        tts_audio = gr.Audio()
        tts_btn.click(fn=synthesize, outputs=tts_audio)

    mic = gr.Audio(sources="microphone", type="filepath")
    output_text = gr.Markdown()
    output_plot = gr.Image()

    run_btn = gr.Button("Analyze")
    run_btn.click(analyze, inputs=mic, outputs=[output_text, output_plot])

demo.launch()
