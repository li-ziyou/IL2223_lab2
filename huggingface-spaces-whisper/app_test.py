import gradio as gr
import os
import sys
import subprocess
import zlib
from typing import Iterator, TextIO
# from moviepy.editor import VideoFileClip

#import whisper
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

#pipe = pipeline(model="tilos/whisper-small-zh-HK")

processor = AutoProcessor.from_pretrained("tilos/whisper-small-zh-HK")
model = AutoModelForSpeechSeq2Seq.from_pretrained("tilos/whisper-small-zh-HK")

model = whisper.load_model("medium")

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"


def translate(input_video):
    audio_file = video2mp3(input_video)

    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio_file, **translate_options) #

    output_dir = '/content/'
    audio_path = audio_file.split(".")[0]

    with open(os.path.join(output_dir, audio_path + ".vtt"), "w") as vtt:
        write_vtt(result["segments"], file=vtt)

    subtitle = audio_path + ".vtt"
    output_video = audio_path + "_subtitled.mp4"

    os.system(f"ffmpeg -i {input_video} -vf subtitles={subtitle} {output_video}")

    return output_video


title = "Add Text/Caption to your YouTube Shorts - MultiLingual"

block = gr.Blocks()

with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style():
                inp_video = gr.Video(
                    label="Input Video",
                    type="filepath",
                    mirror_webcam=False
                )
                op_video = gr.Video()
            btn = gr.Button("Generate Subtitle Video")

        btn.click(translate, inputs=[inp_video], outputs=[op_video])

        gr.HTML()

block.launch(debug=True)