# pip install gradio whisper praat-parselmouth numpy matplotlib pypinyin torch transformers librosa soundfile
# pip install aeneas
#
# for MFA:
# pip install spacy-pkuseg dragonmapper hanziconv
# mfa model download acoustic mandarin_mfa
# mfa model download dictionary mandarin_mfa
# not needed: mfa model download language_model mandarin_mfa_lm
# brew install --cask font-noto-sans-cjk


import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
import subprocess
import parselmouth
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pypinyin import pinyin, Style
from praatio import textgrid
import librosa
import soundfile as sf
import shutil


# https://stackoverflow.com/questions/39630928/how-to-plot-a-figure-with-chinese-characters-in-label
matplotlib.rcParams['font.family'] = ['Heiti TC']


# 1️⃣ Load Whisper
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)

# Correctly set task + language on config:
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

def recognize(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs = inputs.to(DEVICE)
    decoder_prompt = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    model.config.forced_decoder_ids = decoder_prompt

    predicted_ids = model.generate(
        input_features=inputs.input_features
    )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def trim_silence(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    yt, index = librosa.effects.trim(y, top_db=20)  # adjust top_db to taste
    trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
    print(trimmed_path)
    sf.write(trimmed_path, yt, sr)
    return trimmed_path

# 2️⃣ Target phrase
TARGET_TEXT = "我喜欢机器学习"
TARGET_PINYIN = ' '.join(sum(pinyin(TARGET_TEXT, style=Style.TONE), []))

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

TONE_COLORS = {
    '1': '#cce5ff',  # pale blue
    '2': '#ccffcc',  # pale green
    '3': '#fff2cc',  # pale yellow
    '4': '#ffcccc',  # pale red
    '5': '#e0e0e0',  # neutral tone (light grey)
}

def get_tone_color(label):
    import re
    # Extract trailing tone digit
    m = re.search(r'(\d)', label)
    if m:
        return TONE_COLORS.get(m.group(1), '#eeeeee')
    return '#eeeeee'

def plot_alignment_and_pitch(segments, pitch_times, pitch_values, duration):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # word=1, pitch=3

    # Words tier
    ax_words = fig.add_subplot(gs[0])
    for start, end, label in segments:
        color = get_tone_color(label) if label.strip() else "white"
        ax_words.axvspan(start, end, ymin=0.6, ymax=1.0, color=color, alpha=0.3 if label else 0)
        if label.strip():
            ax_words.text((start + end)/2, 0.8, label, ha="center", va="center")

    # Pitch line
    ax_pitch = fig.add_subplot(gs[1], sharex=ax_words)
    ax_pitch.plot(pitch_times, pitch_values / np.nanmax(pitch_values), color="red", lw=2)

    ax_pitch.set_xlim([0, duration])
    ax_pitch.set_ylim([0, 1])
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_yticks([])

    return fig

# 4️⃣ Recognize & analyze
def analyze(audio):
    audio_path = trim_silence(audio)
    hanzi = recognize(audio_path)
    pinyin_out = ' '.join(sum(pinyin(hanzi, style=Style.TONE), []))

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
        "--single_speaker",
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
    fig.tight_layout()
    fig.savefig(plot_path.name, bbox_inches="tight")

    return f"### Whisper Hanzi\n{hanzi}\n\n### Pinyin\n{pinyin_out}\n\n### Target Pinyin\n{TARGET_PINYIN}", plot_path.name, audio_path

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
