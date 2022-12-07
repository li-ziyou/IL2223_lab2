from transformers import pipeline
import gradio as gr
import os
import subprocess

pipe = pipeline(model="tilos/whisper-small-zh-HK")  # change to "your-username/the-name-you-picked"

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

def video_identity(video):
    audio_file = video2mp3(video)
    text = pipe(audio_file)#["text"]
    print(text)
    return text

video = gr.Interface(video_identity,
                    gr.Video(),
                    "playable_video",
                    #examples=[
                    #    os.path.join(os.path.dirname(__file__),
                    #                 "video/video_sample.mp4")],
                    cache_examples=True)

voice = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Cantonese",
    description="Realtime demo for Cantonese speech recognition using a fine-tuned Whisper small model.",
)



demo = gr.TabbedInterface([video, voice])
demo.launch()