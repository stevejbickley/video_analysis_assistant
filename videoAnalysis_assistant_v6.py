##### ------ IMPORTS + SETUP ------- ####

import os
import subprocess
import glob
import ffmpeg
import pandas as pd
import numpy as np
import json
import os
import re
import time
from openai import OpenAI
import openpyxl
import requests
import cv2
import base64
from pydantic import BaseModel

# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_NJ580vd7N4ETnei4zI4LEqlZ" # Old (superseded on 10 April - accidental delete): "asst_wWt15CA9kKqTI79SLQDPlGWm"

# Make sure your API key is set as an environment variable.
client = OpenAI()

############ SETUP
CURR_PATH = os.getcwd()
OUTPUT_PATH = CURR_PATH + '/data/output/'
INPUT_PATH = CURR_PATH + '/data/input/'

##### ------ DEFINE FUNCTIONS ------- ####

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


def frame_by_frame_save(video_path, output_frames_dir='frames'):
    """
    Extracts frames from the video on a frame-by-frame basis.
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    ffmpeg.input(video_path).output(f'{output_frames_dir}/frame_%04d.png').run()
    return output_frames_dir


def frame_by_frame_extraction(video_path, max_frames=250):
    """
    Extracts frames from the video on a frame-by-frame basis and returns a list of (timestamp, frame) tuples,
    ensuring no more than max_frames are extracted. Frames are sampled evenly across the entire video duration.
    """
    frames = []
    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # Calculate the interval to ensure max_frames are extracted
    interval = max(1, frame_count // max_frames)
    # Ensure even distribution of frames across the video duration
    for i in range(max_frames):
        frame_number = min(i * interval, frame_count - 1)  # Ensure we don't go out of bounds
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_number / fps
            frames.append((timestamp, frame))
    cap.release()
    return frames


# Function to extract frames from the video at each second
def extract_frames_by_seconds(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0: # Fallback if fps is 0 or not available
        fps = 24
    duration = frame_count / fps
    for sec in range(int(duration) + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((sec, frame))
        else:
            print(f"Warning: Could not read frame at {sec:.2f} seconds")
    cap.release()
    return frames


def convert_video_with_ffmpeg(input_video_path, output_video_path):
    """
    Converts a video using ffmpeg with specified codecs.
    Args:
    - input_video_path (str): Path to the input video file.
    - output_video_path (str): Path to the output converted video file.
    Returns:
    - None
    """
    # Define the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_video_path
    ]
    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Video converted successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during video conversion: {e}")



def extract_frames_entire_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            if i % int(fps) == 0:  # Grab a frame every second
                frames.append((i / fps, frame))
        else:
            print(f"Warning: Could not read frame at frame number {i}")
    cap.release()
    return frames

def extract_frames_by_frame_number(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, frame_count, int(fps)):  # Grab a frame every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append((i / fps, frame))
        else:
            print(f"Warning: Could not read frame at {i / fps:.2f} seconds")
    cap.release()
    return frames


def encode_frame_to_base64(frame):
    """
    Encodes a single frame (image) to base64 format.
    """
    _, buffer = cv2.imencode('.png', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    return encoded_frame


def prepare_frames_for_api(frames):
    """
    Prepares a list of frames (timestamps, images) for the OpenAI API.
    Encodes each frame in base64 and creates the payload for the API.
    """
    prepared_frames = []
    for timestamp, frame in frames:
        encoded_frame = encode_frame_to_base64(frame)
        prepared_frames.append({
            "timestamp": timestamp,
            "encoded_frame": encoded_frame
        })
    return prepared_frames


json_schema = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "The time when the event occurs"
            }
        },
        "activity_event_action": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A description of the activity, event, or action that occurs at the timestamp"
            }
        },
        "scene_background_context": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "The context or setting of the scene at the timestamp"
            }
         },
        "behavioural_nudge_device_or_moral_suasion": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Any behavioural nudge, device, or moral suasion used in the scene"
            }
        }
    },
    "required": ["timestamp", "activity_event_action", "scene_background_context", "behavioural_nudge_device_or_moral_suasion"],
    "additionalProperties": False
}



def analyze_frames_with_openai_jsonSchema(prepared_frames, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    for frame in prepared_frames:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{frame['encoded_frame']}"
            }
        })
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "StoryboardExtractionFromVideo",
                "schema": json_schema,
                "strict": True
            }
        },
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()



class StoryboardExtractionFromVideo(BaseModel):
    timestamp: list[str]
    activity_event_action: list[str]
    scene_background_context: list[str]
    behavioural_nudge_device_or_moral_suasion: list[str]



def analyze_frames_with_openai_sdk(prepared_frames, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    for frame in prepared_frames:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{frame['encoded_frame']}"
            }
        })
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "StoryboardExtractionFromVideo2",
                "schema": StoryboardExtractionFromVideo.model_json_schema()
            }
        },
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def analyze_frames_with_openai(prepared_frames, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    for frame in prepared_frames:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{frame['encoded_frame']}"
            }
        })
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()



def generate_event_log_or_storyboard_unstructured(frames, prompt, api_key, model="gpt-4o-mini"):
    """
    Generates an event log or storyboard based on the video frames and the given prompt.
    """
    prepared_frames = prepare_frames_for_api(frames)
    response = analyze_frames_with_openai(prepared_frames, prompt, api_key, model)
    event_log = []
    # Parse the response to extract and structure the event log/storyboard
    for i, frame in enumerate(prepared_frames):
        event_description = response['choices'][0]['message']['content'].split('\n')[i]
        event_log.append({
            "Timestamp": f"{frame['timestamp']:.1f}s",
            "Description": event_description
        })
    return event_log


def parse_response_content(content):
    """
    Parses the JSON-like string output from the response and structures it into a list of dictionaries.
    Args:
    - content (str): The JSON-like string from the API response.
    Returns:
    - List[Dict]: A list of dictionaries with the structured data.
    """
    # Parse the JSON string into a dictionary
    data = json.loads(content)
    # Converting json dataset from dictionary to dataframe
    df = pd.DataFrame.from_dict(data)
    df.reset_index(inplace=True)
    return df


def generate_event_log_or_storyboard_structuredOutput(frames, prompt, api_key, model="gpt-4o-mini"):
    """
    Generates an event log or storyboard based on the video frames and the given prompt.
    """
    prepared_frames = prepare_frames_for_api(frames)
    response = analyze_frames_with_openai_sdk(prepared_frames, prompt, api_key, model)
    #response = analyze_frames_with_openai_jsonSchema(prepared_frames, prompt, api_key, model)
    event_log = parse_response_content(response['choices'][0]['message']['content'])
    return event_log


def check_if_all_arrays_sum_to_zero(frames):
    """
    Checks if all arrays in a list of arrays (frames) sum to zero.
    Args:
    - frames (list of list of int/float): A list of arrays/lists containing numeric values.
    Returns:
    - bool: True if all arrays sum to zero, False otherwise.
    """
    for timestamp, array in frames:
        if np.sum(array) != 0:
            return False
    return True


##### ------ DEFINE FUNCTIONS - END ------- ####

#EXAMPLES
#response = analyze_frames_with_openai_jsonSchema(prepared_frames, prompt, os.environ['OPENAI_API_KEY'],model = "gpt-4o-2024-08-06")
#responseparsed = parse_response_content(response['choices'][0]['message']['content'])
#response2 = analyze_frames_with_openai_sdk(prepared_frames, prompt, os.environ['OPENAI_API_KEY'],model = "gpt-4o-2024-08-06")
#response2parsed = parse_response_content(response2['choices'][0]['message']['content'])

# Find all ".mp4" files in the input file
allfiles = glob.glob(INPUT_PATH + '*.mp4')

# Process each video file one-by-one
for video_path in allfiles:
    if os.path.exists(OUTPUT_PATH + video_path.split('/')[-1][:-4] + 'event_log.csv'):  # CHANGE for euro / champions / europa / world cup
        continue
    # Extract frames from the video
    #frames = extract_frame_by_frame(video_path) # old/superseded code
    frames = extract_frames_by_seconds(video_path)
    if check_if_all_arrays_sum_to_zero(frames):
        frames = extract_frames_entire_video(video_path)
        if check_if_all_arrays_sum_to_zero(frames):
            convert_video_with_ffmpeg(video_path, video_path[:-4]+'_converted.mp4')
            frames = extract_frames_by_seconds(video_path[:-4]+'_converted.mp4')
            if check_if_all_arrays_sum_to_zero(frames):
                print('Video file path failed to extract frames: ' + video_path)
                continue
    # Prompt for OpenAI API
    prompt = "Please review the images and then provide me a full historical action/activity/event log of everything that happens at each time stamp in a structured table as well as scene/background context too, including any behavioural nudge, device, or moral suasion used."
    # Generate the event log
    try:
        event_log = generate_event_log_or_storyboard_structuredOutput(frames, prompt, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06") # Output columns: ["timestamp", "activity_event_action", "scene_background_context", "behavioural_nudge_device_or_moral_suasion"]
    except:
        print('Video file path failed to extract event log: '+video_path)
        continue
    # Save to CSV
    event_log.to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + 'event_log.csv', index=False)
    # Print the event log
    print(event_log)

