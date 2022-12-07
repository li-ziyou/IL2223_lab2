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

def audio_preprocess(audio_file):
    filename, ext = os.path.splitext(audio_file)
    return f"{filename}.{ext}"

def offline_audio(audio):
    audio_file = audio_preprocess(audio)
    text = transcribe(audio_file)
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

def offline_video(video):
    audio_file = video2mp3(video)
    text = transcribe(audio_file)
    return text


with gr.Blocks() as demo:
    # audio file input
    gr.Interface(
            fn=offline_audio,
            title="Whisper: zh-HK Subtitle Generator",
            description="Generate zh-HK subtitle from audio file, your microphone and Youtube",
            inputs = "audio",
            outputs = "text",
            allow_flagging= "never",
    )
    # video file input
    gr.Interface(
            fn=offline_video,
            inputs="video",
            outputs="text",
            allow_flagging="never",
        )

    # microphone input
    with gr.Row():
        with gr.Column():
            input_mircro = gr.Audio(source="microphone", type="filepath")
            micro_btn = gr.Button('Generate Voice Subtitles')
        with gr.Column():
            output_micro = gr.Textbox(placeholder='Transcript from mic', label='Subtitles')
            micro_btn.click(transcribe, inputs=input_mircro, outputs=output_micro)

    # Youtube url input
    with gr.Row():
        with gr.Column():
            inputs_url = gr.Textbox(placeholder='Youtube URL', label='URL')
            url_btn = gr.Button('Generate Youtube Video Subtitles')
            examples = gr.Examples(examples=["https://www.youtube.com/watch?v=Yw4EoGWe0vw"],inputs=[inputs_url])
        with gr.Column():
            output_url = gr.Textbox(placeholder='Transcript from video.', label='Transcript')
            url_btn.click(get_text, inputs=inputs_url, outputs=output_url )



demo.launch()