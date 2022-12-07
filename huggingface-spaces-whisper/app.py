from transformers import pipeline
import gradio as gr
import os
import subprocess
from pytube import YouTube

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

def get_text(url):
    if url != '': output_text_transcribe = ''
    result = pipe(get_audio(url))
    return result['text'].strip()

def get_audio(url):
    website = YouTube(url)
    video = website.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=".")
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    audio = new_file
    return audio

def video_identity(video):
    audio_file = video2mp3(video)
    text = pipe(audio_file)#["text"]
    print(text)
    return text
def test(a):
    return a

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_mircro = gr.Audio(source="microphone", type="filepath")
            micro_btn = gr.Button('Transcribe voice')
        with gr.Column():
            output_micro = gr.Textbox(placeholder='Transcript from mic.', label='Transcript') #label?
            micro_btn.click(transcribe, inputs=input_mircro, outputs=output_micro)

    with gr.Row():
        with gr.Column():
            inputs_url = gr.Textbox(placeholder='Video URL', label='URL')
            url_btn = gr.Button('Transcribe Video')
        with gr.Column():
            output_url = gr.Textbox(placeholder='Transcript from video.', label='Transcript')
            url_btn.click(get_text, inputs=inputs_url, outputs=output_url)
            examples = [["www.google.com"]]
    with gr.Row():
        gr.Interface(
            fn=test,
            inputs="audio",
            outputs = "audio",
            examples=[]
        )

demo.launch()