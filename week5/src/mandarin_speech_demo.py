# pip install gradio whisper praat-parselmouth numpy matplotlib pypinyin torch transformers librosa
#
# for MFA:
# pip install spacy-pkuseg dragonmapper hanziconv
# mfa model download acoustic mandarin_mfa
# mfa model download dictionary mandarin_mfa
# not needed: mfa model download language_model mandarin_mfa_lm


import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
import subprocess
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from pypinyin import pinyin, Style
from praatio import textgrid
import librosa
import shutil


# 1️⃣ Load Whisper
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)

# Correctly set task + language on config:
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

def recognize(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs = inputs.to(DEVICE)
    predicted_ids = model.generate(
        input_features=inputs.input_features
    )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# 2️⃣ Target phrase
TARGET_TEXT = "我喜欢机器学习"
TARGET_PINYIN = ' '.join(sum(pinyin(TARGET_TEXT, style=Style.TONE3), []))

# 3️⃣ MFA setup
MFA_MODEL = "mandarin_mfa"  # download from MFA site
MFA_DICT = os.path.expanduser("/Users/anton/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict")

# TTS: optional, use Coqui or Bark here — or skip for now
def synthesize():
    return "demo_tts_output.wav"

def parse_textgrid(tg_path):
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    words_tier = tg.getTier("words")
    intervals = words_tier.entries
    return intervals  # List of (start, end, label)

def plot_alignment_and_pitch(segments, pitch_times, pitch_values, duration):
    fig, ax = plt.subplots(figsize=(12, 3))

    # Words tier
    for start, end, label in segments:
        color = "blue" if label.strip() else "white"
        ax.axvspan(start, end, ymin=0.6, ymax=1.0, color=color, alpha=0.3 if label else 0)
        if label.strip():
            ax.text((start + end)/2, 0.8, label, ha="center", va="center")

    # Pitch line
    ax.plot(pitch_times, pitch_values / np.nanmax(pitch_values), color="red", lw=2)

    ax.set_xlim([0, duration])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])

    return fig

# 4️⃣ Recognize & analyze
def analyze(audio):
    audio_path = audio
    hanzi = recognize(audio_path)
    pinyin_out = ' '.join(sum(pinyin(hanzi, style=Style.TONE3), []))

    # MFA: same as before
    corpus_dir = tempfile.mkdtemp()
    corpus_audio = os.path.join(corpus_dir, "utt1.wav")
    shutil.copy(audio_path, corpus_audio)
    text_path = os.path.join(corpus_dir, "utt1.lab")
    with open(text_path, "w") as f:
        f.write(TARGET_TEXT)
    mfa_output_dir = tempfile.mkdtemp()
    subprocess.run([
        "mfa", "align",
        corpus_dir,
        MFA_DICT,
        MFA_MODEL,
        mfa_output_dir
    ])

    # ✅ Load TextGrid
    tg_path = os.path.join(mfa_output_dir, "utt1.TextGrid")
    segments = parse_textgrid(tg_path)

    # ✅ Pitch
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_times = pitch.xs()
    duration = snd.duration

    # ✅ Plot combined
    fig = plot_alignment_and_pitch(segments, pitch_times, pitch_values, duration)
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

    mic = gr.Audio(sources=["microphone", "upload"], type="filepath")
    output_text = gr.Markdown()
    output_plot = gr.Image()

    run_btn = gr.Button("Analyze")
    run_btn.click(analyze, inputs=mic, outputs=[output_text, output_plot])

demo.launch()
