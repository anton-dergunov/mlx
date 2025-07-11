import gradio as gr
import random
import pandas as pd

VIDEO_MAP = {
    "Emotional damage": "videos/emotional_damage.mp4",
    "Chocolate": "videos/chocolate.mp4",
    "It do go down": "videos/it_do_go_down.mp4",
    "Surprise surprise": "videos/surprise_surprise.mp4",
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


def predict_emotions(video_choice):
    probs = [random.uniform(0, 1) for _ in emotion_map]
    total = sum(probs)
    probs = [p / total for p in probs]

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

demo.launch()
