##### ------ IMPORT FUNCTIONS + SETUP CODE - START ------- ####

import ffmpeg
import openai
import os
from pydub import AudioSegment
import numpy as np
#import librosa
import json
import os
import re
import time
import pandas as pd
from openai import OpenAI
import openpyxl
import requests
from pyannote.audio import Pipeline
import cv2


# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_NJ580vd7N4ETnei4zI4LEqlZ" # Old (superseded on 10 April - accidental delete): "asst_wWt15CA9kKqTI79SLQDPlGWm"

# Make sure your API key is set as an environment variable.
client = OpenAI()

##### ------ IMPORT FUNCTIONS + SETUP CODE - END ------- ####

##### ------ DEFINE FUNCTIONS - START ------- ####

def create_run(assistant_id, thread_id, message_content):
    # Add message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_content,
    )
    # Create the run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return run

def wait_on_run(run, thread_id):
    while run['status'] in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run['id'],
        )
        time.sleep(0.5)
    return run


def extract_audio(video_path, output_audio_path='output_audio.wav'):
    """
    Extracts audio from the given MPEG-4 video file.
    """
    ffmpeg.input(video_path).output(output_audio_path).run()
    return output_audio_path


def frame_by_frame_extraction(video_path, output_frames_dir='frames'):
    """
    Extracts frames from the video on a frame-by-frame basis.
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    ffmpeg.input(video_path).output(f'{output_frames_dir}/frame_%04d.png').run()
    return output_frames_dir


def transcribe_audio(audio_path, model="whisper-1", language=None, response_format="verbose_json", timestamp_granularities=["segment"]):
    """
    Transcribes audio using OpenAI's API.
    Args:
        audio_path (str): The path to the audio file.
        model (str): The model ID to use for transcription. Default is "whisper-1".
        language (str): The language of the input audio in ISO-639-1 format (e.g., 'en' for English). Optional.
        response_format (str): The format of the transcript output. Defaults to "verbose_json".
        timestamp_granularities (list): The timestamp granularities for the transcription. Defaults to ["segment"].
    Returns:
        dict: The transcription object (verbose JSON format).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/audio/transcriptions"
    with open(audio_path, 'rb') as audio_file:
        files = {
            'file': audio_file
        }
        data = {
            'model': model,
            'response_format': response_format,
            'timestamp_granularities[]': timestamp_granularities
        }
        if language:
            data['language'] = language
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'multipart/form-data'
        }
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Transcription failed with status code {response.status_code}: {response.text}")


def speaker_diarization(audio_path):
    """
    Perform speaker diarization using Pyannote's pre-trained model.
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
    diarization = pipeline({'uri': 'blabal', 'audio': audio_path})
    # Save the diarization output to a text file
    with open("diarization.txt", "w") as text_file:
        text_file.write(str(diarization))
    # Print out diarization segments
    print(*list(diarization.itertracks(yield_label=True))[:10], sep="\n")
    return diarization


def align_transcription_with_frames(transcript, frames_dir):
    """
    Align the transcription with corresponding video frames.
    """
    # Placeholder: Implement logic to sync transcript timings with frame timestamps
    alignment = {}
    for frame in sorted(os.listdir(frames_dir)):
        alignment[frame] = {"text": transcript, "speaker": "Unknown"}
    return alignment


def millisec(timeStr):
    spl = timeStr.split(":")
    return int((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)


def process_diarization(diarization_file):
    """
    Process the diarization text file and return a list of segments.
    """
    dz = open(diarization_file).read().splitlines()
    dzList = []
    for l in dz:
        start, end = tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
        start = millisec(start)
        end = millisec(end)
        speaker = 'SPEAKER_01' in l
        dzList.append([start, end, speaker])
    return dzList


def generate_html(dzList, transcript, frames_dir):
    """
    Generate an HTML file that aligns transcription with speaker diarization.
    """
    # Prepare your HTML structure
    preS = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Transcription</title>\n<style>\nbody { font-family: sans-serif; font-size: 18px; color: #111; padding: 0 0 1em 0; }\n.l { color: #050; }\n.s { display: inline-block; }\n.e { display: inline-block; }\n.t { display: inline-block; }\n#player { position: sticky; top: 20px; float: right; }\n</style>\n</head>\n<body>\n<div id="player"></div>\n'
    postS = '</body>\n</html>'
    html = [preS]
    spacer_milli = 0  # adjust for any spacer if needed
    for i, segment in enumerate(dzList):
        start_str = '{:02d}:{:02d}:{:05.2f}'.format(segment[0] // 3600, (segment[0] % 3600) // 60, segment[0] % 60)
        html.append(
            f'<div class="c">\n<a class="l" href="#{start_str}" id="{start_str}">link</a> |\n<div class="s"><a href="javascript:void(0);" onclick="setCurrentTime({segment[0] / 1000})">{start_str}</a></div>\n<div class="t">{"[Lex]" if segment[2] else "[Yann]"} {transcript}</div>\n</div>\n')
    html.append(postS)
    with open("transcription.html", "w") as text_file:
        text_file.write(''.join(html))


def process_video(video_path):
    """
    Full processing pipeline for a given video file.
    """
    audio_path = extract_audio(video_path)
    frames_dir = frame_by_frame_extraction(video_path)
    transcript = transcribe_audio(audio_path)
    diarization = speaker_diarization(audio_path)
    # Process diarization results
    dzList = process_diarization("diarization.txt")
    # Generate an aligned HTML transcription
    generate_html(dzList, transcript, frames_dir)
    return frames_dir, dzList


def process_video_with_assistant(video_path):
    # Step 1: Create a new thread for this video
    thread = client.beta.threads.create()
    thread_id = thread['id']
    print(f"Created Thread with ID: {thread_id} for video {video_path}")
    # Step 1: Extract Audio
    audio_path = extract_audio(video_path)
    # Step 2: Frame-by-frame extraction
    frames_dir = frame_by_frame_extraction(video_path)
    # Step 3: Start a new run with a message to the assistant
    run = create_run(assistant_id, thread_id, f"Process the video at {video_path}")
    # Step 4: Wait for the run to complete
    run = wait_on_run(run, thread_id)
    # Step 5: Retrieve and print results
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for message in messages['data']:
        print(f"{message['role']}: {message['content'][0]['text']['value']}")
    return audio_path, frames_dir


##### ------ DEFINE FUNCTIONS - END ------- ####

video_path = "Shannons 'Whatever You Ride. Ride with Shannons' 30sec Commercial.mp4"

# Example usage
video_path = "example.mp4"
process_video_with_assistant(video_path)

# Example usage
video_path = "Panalogy_Lab_meeting_recording.mp4"
frames_dir, dzList = process_video(video_path)

# Output the aligned transcription
for frame, data in result.items():
    print(f"{frame}: {data['speaker']} said '{data['text']}'")


audio_path = '/Users/stevenbickley/stevejbickley/videoAnalysis_assistant/output_audio.wav'
transcription = transcribe_audio(audio_path)

