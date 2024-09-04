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
import io
from pydub import AudioSegment
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


def extract_audio(video_path, output_audio_path='output_audio.mp3', save=True):
    """
    Extracts audio from the given MPEG-4 video file.
    If save is True, saves the audio to the specified output path.
    If save is False, returns the audio file as an MP3 in-memory file.
    """
    if save: # Run ffmpeg to extract audio and save to file if save is True
        ffmpeg.input(video_path).output(output_audio_path, format='mp3').run()
        return output_audio_path
    else:
        # Extract audio to an in-memory buffer
        output, stderr = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='wav')
            .run(capture_stdout=True)
        )
        # If there was an error then print it
        if stderr:
            print("FFmpeg encountered the following warnings/errors:", stderr.decode())
        # Convert WAV to MP3 using pydub
        audio = AudioSegment.from_wav(io.BytesIO(output))
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3")
        mp3_io.seek(0)  # Reset the pointer to the start of the file
        return mp3_io


def transcribe_audio(video_path, segment_timing=True):
    # Assume mp3_io is returned from your extract_audio function with save=False
    mp3_io = extract_audio(video_path, save=False)
    # Reset the file pointer to the beginning
    mp3_io.seek(0)
    # Set a filename (as the API expects an actual filename, even when using an in-memory object)
    mp3_io.name = "audio.mp3"  # Set a name attribute for the in-memory BytesIO object
    # Transcribe the in-memory MP3 file using the Whisper API
    if segment_timing:
        transcription = client.audio.transcriptions.create(
            model = "whisper-1",
            file=mp3_io,
            response_format="verbose_json",
            prompt="Hello, welcome to my lecture." # Sometimes the model might skip punctuation in the transcript
        )
    else:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=mp3_io,
            response_format="verbose_json",
            prompt="Hello, welcome to my lecture.",  # Sometimes the model might skip punctuation in the transcript
            timestamp_granularities = ["word"]  # Defaults to segment
        )
    # Finally, return the transcribe object
    return transcription


def extract_segments(transcription, full_report=False):
    """
    Extracts the segments from the transcription and returns a structured dataframe.
    Args:
    - transcription: A dictionary containing the transcription result including the segments.
    Returns:
    - A pandas DataFrame containing the segment ID, start time, end time, and text.
    """
    segments = transcription.segments
    # Extract relevant information from each segment
    data = []
    for segment in segments:
        if full_report:
            data.append({
                'id': segment['id'],
                'seek': segment['seek'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'tokens': segment['tokens'],
                'temperature': segment['temperature'],
                'avg_logprob': segment['avg_logprob'],
                'compression_ratio': segment['compression_ratio'],
                'no_speech_prob': segment['no_speech_prob']
            })
        else:
            data.append({
                'id': segment['id'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)
    return df



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
            "description": "Timestamps for each event",
            "items": {
                "type": "string"
            }
        },
        "activity_event_action": {
            "type": "array",
            "description": "A description of the event/activity happening at the time",
            "items": {
                "type": "string"
            }
        },
        "scene_background_context": {
            "type": "array",
            "description": "The context/setting of the background scene at the time",
            "items": {
                "type": "string"
            }
         },
        "behavioural_nudge_device_or_moral_suasion": {
            "type": "array",
            "description": "Any behavioural nudges, devices, or moral suasion used in the scene",
            "items": {
                "type": "string",
            }
        },
        "hidden_messages": {
            "type": "array",
            "description": "Any subtle/implicit messages that require 'reading between the lines' in the scene",
            "items": {
                "type": "string"
            }
        },
        "humour": {
            "type": "array",
            "description": "Presence of humor or comedic elements in the scene",
            "items": {
                "type": "string"
            }
        },
        "overall_summary": {
            "type": "string",
            "description": "Any behavioural nudges, devices, or moral suasion used in the scene"
            }
        }
    },
    "required": ["timestamp", "activity_event_action", "scene_background_context", "behavioural_nudge_device_or_moral_suasion", "hidden_messages", "humour", "overall_summary"],
    "additionalProperties": False
}



def analyze_frames_with_openai_jsonSchema(prepared_frames, transcription, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Append the transcription information directly to the prompt
    full_prompt = f"{prompt}\nThe audio transcription is: {transcription.to_json(orient='records', lines=False)}"
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "You are a helpful assistant that analyzes frames and the audio transcript from an online video/television ad. Generate a structured action/event log with context from both video and audio for each timestamp, then provide an overall summary.Output in JSON format."}]},
        {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
    ]
    for frame in prepared_frames:
        messages[1]["content"].append({
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
    timestamp: list[str] # Timestamps for each event
    activity_event_action: list[str] # Description of the event/activity happening at the time
    scene_background_context: list[str] # Context of the background scene
    behavioural_nudge_device_or_moral_suasion: list[str]  # Nudges, devices, or moral suasion
    hidden_messages: list[str] # Any subtle/implicit messages that require 'reading between the lines'
    humour: list[str] # Presence of humor or comedic elements
    overall_summary: str # Qualitative summary of the ad at the end - In future: more into the shape of the story of movies/books... overall analysis or understanding or summary of the video as a whole


def analyze_frames_with_openai_sdk(prepared_frames, transcription, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Append the transcription information directly to the prompt
    full_prompt = f"{prompt}\nThe audio transcription is: {transcription.to_json(orient='records', lines=False)}"
    messages = [
        {"role": "system","content": "Analyze frames from an online video/television ad. Generate a structured action log with relevant context for each timestamp and provide an overall summary. Output in JSON format."},
        {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
    ]
    for frame in prepared_frames:
        messages[1]["content"].append({
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


def analyze_frames_with_openai(prepared_frames, transcription, prompt, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Append the transcription information directly to the prompt
    full_prompt = f"{prompt}\nThe audio transcription is: {transcription.to_json(orient='records', lines=False)}"
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "You are a helpful assistant that analyzes frames and the audio transcript from an online video/television ad. Generate a structured action/event log with context from both video and audio for each timestamp, then provide an overall summary.Output in JSON format."}]},
        {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
    ]
    for frame in prepared_frames:
        messages[1]["content"].append({
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
    try:
        data = json.loads(content)
    except:
        data = content
    # Converting json dataset from dictionary to dataframe
    df = pd.DataFrame.from_dict(data)
    df.reset_index(inplace=True)
    return df


def generate_event_log_or_storyboard_structuredOutput(video_path, frames, transcription, prompt, api_key, model="gpt-4o-mini"):
    """
    Generates an event log or storyboard based on the video frames and the given prompt.
    """
    prepared_frames = prepare_frames_for_api(frames)
    response = analyze_frames_with_openai_sdk(prepared_frames, transcription, prompt, api_key, model)
    #response = analyze_frames_with_openai_jsonSchema(prepared_frames, transcription, prompt, api_key, model)
    event_log = parse_response_content(response['choices'][0]['message']['content'])
    # Save the response details to CSV
    save_response_outputs(response, video_path, filename='videoLog')
    # Assuming biases_tempo is a list of dictionaries (one for each bias entry)
    selected_columns = ['id', 'object', 'created', 'model', 'usage']
    # Ensure that only dictionaries are processed
    filtered_data = [list([k, response[k]]) for k in ['id', 'object', 'created', 'model', 'usage']]
    # Convert the filtered data into a pandas DataFrame
    event_log_details = pd.DataFrame(filtered_data)
    # Save to CSV
    event_log_details.to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + '_videoLog_metadata.csv', index=False)
    # Save to CSV
    pd.DataFrame.from_dict(response['choices']).to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + '_videoLog_detailedResponse.csv', index=False)
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


# Full list of biases and heuristics
biases_heuristics = [
    {
        "Name of Bias": "Availability Heuristic",
        "Category of Bias": "Information",
        "Description": "Memories that happened recently outweigh memories with more impact from the past."
    },
    {
        "Name of Bias": "Attentional Bias",
        "Category of Bias": "Information",
        "Description": "Subconsciously we choose points where we pay attention to. A smoker is more likely to notice other people smoking."
    },
    {
        "Name of Bias": "Illusory Truth Effect",
        "Category of Bias": "Information",
        "Description": "By repeatedly hearing false information, we come to believe it is the truth."
    },
    {
        "Name of Bias": "Mere Exposure Effect",
        "Category of Bias": "Information",
        "Description": "We give preference to things we have seen more often or are familiar with, affecting the value we place on them."
    },
    {
        "Name of Bias": "Context Effect",
        "Category of Bias": "Information",
        "Description": "The context in which we see something affects how we look at it and the value we place on it."
    },
    {
        "Name of Bias": "Cue-Dependent Forgetting",
        "Category of Bias": "Information",
        "Description": "Remembering certain things by thinking of equivalent memories."
    },
    {
        "Name of Bias": "Frequency Illusion / Baader-Meinhof Phenomenon",
        "Category of Bias": "Information",
        "Description": "If you see something once and start paying attention to it, you will see it more often (e.g., buying a car and seeing the same model more often)."
    },
    {
        "Name of Bias": "Empathy Gap",
        "Category of Bias": "Information",
        "Description": "We are bad at empathizing with others but expect others to empathize with us."
    },
    {
        "Name of Bias": "Von Restorff Effect",
        "Category of Bias": "Information",
        "Description": "We remember one striking thing in a series of equivalent things better."
    },
    {
        "Name of Bias": "Bizarreness Effect",
        "Category of Bias": "Information",
        "Description": "We remember bizarre things better than normal things."
    },
    {
        "Name of Bias": "Humor Effect",
        "Category of Bias": "Information",
        "Description": "We remember humorous or funny things better than non-humorous things."
    },
    {
        "Name of Bias": "Self-Reference Effect",
        "Category of Bias": "Information",
        "Description": "We remember things we feel related to or connected with better."
    },
    {
        "Name of Bias": "Picture Superiority Effect",
        "Category of Bias": "Information",
        "Description": "We remember pictures better than text."
    },
    {
        "Name of Bias": "Negativity Bias",
        "Category of Bias": "Information",
        "Description": "People are more likely to be attracted to negativity."
    },
    {
        "Name of Bias": "Anchoring",
        "Category of Bias": "Information",
        "Description": "We hammer down on the first piece of information we receive, interpreting new information based on the first."
    },
    {
        "Name of Bias": "Conservatism Bias",
        "Category of Bias": "Information",
        "Description": "We do not sufficiently revise our opinion when we're shown new evidence."
    },
    {
        "Name of Bias": "Contrast Effect",
        "Category of Bias": "Information",
        "Description": "Red dots on a white background appear black, but on a gray background, they appear red."
    },
    {
        "Name of Bias": "Distinction Bias",
        "Category of Bias": "Information",
        "Description": "We place more value on the difference between two options when comparing them than we would if evaluating them separately."
    },
    {
        "Name of Bias": "Framing Effect",
        "Category of Bias": "Information",
        "Description": "People are influenced by how information is presented, e.g., 20% fat vs. 80% fat-free."
    },
    {
        "Name of Bias": "Confirmation Bias / Continued Influence",
        "Category of Bias": "Information",
        "Description": "We unconsciously focus on things that align with our beliefs."
    },
    {
        "Name of Bias": "Post-Purchase Rationalization / Choice-Supportive Bias",
        "Category of Bias": "Information",
        "Description": "After making a choice, we overvalue the positive aspects and ignore the negatives."
    },
    {
        "Name of Bias": "Selective Perception",
        "Category of Bias": "Information",
        "Description": "We ignore certain things if they don't align with our beliefs."
    },
    {
        "Name of Bias": "Observer-Expectancy Effect",
        "Category of Bias": "Information",
        "Description": "People behave differently when they are being observed."
    },
    {
        "Name of Bias": "Experimenter’s Bias / Observer Effect",
        "Category of Bias": "Information",
        "Description": "A researcher can unconsciously influence participants to behave in a certain way."
    },
    {
        "Name of Bias": "Ostrich Effect",
        "Category of Bias": "Information",
        "Description": "Ignoring negative things like feedback or criticism, 'burying your head in the sand.'"
    },
    {
        "Name of Bias": "Clustering Illusion",
        "Category of Bias": "Meaning",
        "Description": "We think we see clusters or patterns in randomly distributed objects."
    },
    {
        "Name of Bias": "Semmelweis Reflex",
        "Category of Bias": "Information",
        "Description": "Rejecting new evidence because it doesn't match current norms, beliefs, or paradigms."
    },
    {
        "Name of Bias": "Naïve Cynicism",
        "Category of Bias": "Information",
        "Description": "We think people are more selfish than they really are."
    },
    {
        "Name of Bias": "Naïve Realism / Bias Blind Spot",
        "Category of Bias": "Information",
        "Description": "We think we see the world objectively and others who don’t share our view are misinformed, irrational, or biased."
    },
    {
        "Name of Bias": "Neglect of Probability",
        "Category of Bias": "Meaning",
        "Description": "Neglecting small chances of something happening. E.g., not wearing a seatbelt because accidents are unlikely."
    },
    {
        "Name of Bias": "Insensitivity to Sample Size",
        "Category of Bias": "Meaning",
        "Description": "Not considering the size of a sample when interpreting data. Variability is greater in smaller sample sizes."
    },
    {
        "Name of Bias": "Anecdotal Fallacy",
        "Category of Bias": "Meaning",
        "Description": "Relying on personal experience as evidence, even when real evidence contradicts it."
    },
    {
        "Name of Bias": "Illusion of Validity",
        "Category of Bias": "Meaning",
        "Description": "We overestimate our ability to make judgments or decisions based on data."
    },
    {
        "Name of Bias": "Gambler's Fallacy",
        "Category of Bias": "Meaning",
        "Description": "Believing that a certain event is less likely to happen if it has occurred frequently in the past."
    },
    {
        "Name of Bias": "Hot-Hand Fallacy",
        "Category of Bias": "Meaning",
        "Description": "Thinking that success in the past will lead to future success, regardless of other factors."
    },
    {
        "Name of Bias": "Illusory Correlation",
        "Category of Bias": "Meaning",
        "Description": "Perceiving a relationship between two variables when none exists."
    },
    {
        "Name of Bias": "Pareidolia",
        "Category of Bias": "Meaning",
        "Description": "Seeing patterns or objects that aren't actually there, e.g., seeing shapes in clouds."
    },
    {
        "Name of Bias": "Anthropomorphism",
        "Category of Bias": "Meaning",
        "Description": "Attributing human traits, emotions, or intentions to non-human entities."
    },
    {
        "Name of Bias": "Group Attribution Error",
        "Category of Bias": "Meaning",
        "Description": "Assuming that the behavior or characteristics of an individual reflect those of the entire group."
    },
    {
        "Name of Bias": "Stereotyping",
        "Category of Bias": "Meaning",
        "Description": "Holding a generalized belief about a particular group of people."
    },
    {
        "Name of Bias": "Essentialism",
        "Category of Bias": "Meaning",
        "Description": "Believing that certain properties are essential to the identity of an object or entity."
    },
    {
        "Name of Bias": "Functional Fixedness",
        "Category of Bias": "Meaning",
        "Description": "Using objects only for their traditional or intended purpose, limiting creativity."
    },
    {
        "Name of Bias": "Self-licensing",
        "Category of Bias": "Meaning",
        "Description": "Increased self-confidence can lead to engaging in undesirable behavior, believing there are no consequences."
    },
    {
        "Name of Bias": "Just-World Hypothesis",
        "Category of Bias": "Meaning",
        "Description": "Belief that the world is inherently fair, and people get what they deserve—good things happen to good people, and bad things happen to bad people."
    },
    {
        "Name of Bias": "Authority Bias",
        "Category of Bias": "Meaning",
        "Description": "We value the opinion of an authority figure more highly and are more likely to follow them without questioning."
    },
    {
        "Name of Bias": "Automation Bias",
        "Category of Bias": "Meaning",
        "Description": "We trust automated systems or suggestions, even when non-automated options may be better."
    },
    {
        "Name of Bias": "Bandwagon Effect",
        "Category of Bias": "Meaning",
        "Description": "Adopting certain behaviors, beliefs, or styles because others are doing it."
    },
    {
        "Name of Bias": "Placebo Effect",
        "Category of Bias": "Meaning",
        "Description": "Believing that something works is often enough to make it effective, even if it has no therapeutic properties."
    },
    {
        "Name of Bias": "Cross-Race Effect",
        "Category of Bias": "Meaning",
        "Description": "We recognize faces of people from our own race better than faces of people from other races."
    },
    {
        "Name of Bias": "In-Group Favoritism",
        "Category of Bias": "Meaning",
        "Description": "We show preference for people in our own group rather than those outside the group."
    },
    {
        "Name of Bias": "Halo Effect",
        "Category of Bias": "Meaning",
        "Description": "Judging people, companies, or products based on a single positive trait or past performance, rather than the present reality."
    },
    {
        "Name of Bias": "Cheerleader Effect",
        "Category of Bias": "Meaning",
        "Description": "People seem more attractive when they are in a group than when they are alone."
    },
    {
        "Name of Bias": "Positivity Effect",
        "Category of Bias": "Meaning",
        "Description": "We tend to focus on positive news or outcomes more than negative ones."
    },
    {
        "Name of Bias": "Not Invented Here",
        "Category of Bias": "Meaning",
        "Description": "Dismissing ideas or products because they come from external sources, favoring one's own ideas."
    },
    {
        "Name of Bias": "Reactive Devaluation",
        "Category of Bias": "Meaning",
        "Description": "Undervaluing a suggestion or solution simply because it comes from someone who is seen as different or opposed to you."
    },
    {
        "Name of Bias": "Well-Traveled Road Effect",
        "Category of Bias": "Meaning",
        "Description": "Familiar routes seem shorter in time than unfamiliar routes."
    },
    {
        "Name of Bias": "Mental Accounting",
        "Category of Bias": "Meaning",
        "Description": "We value money differently depending on how it was earned or gained."
    },
    {
        "Name of Bias": "Appeal to Probability Fallacy / Murphy's Law",
        "Category of Bias": "Meaning",
        "Description": "Assuming that if something has a chance of happening, it will eventually happen."
    },
    {
        "Name of Bias": "Normalcy Bias",
        "Category of Bias": "Meaning",
        "Description": "Underestimating or ignoring the likelihood of disaster or negative events because they haven't happened before."
    },
    {
        "Name of Bias": "Zero-Sum Bias",
        "Category of Bias": "Meaning",
        "Description": "Believing that if one person gains, another must lose, assuming the world is zero-sum."
    },
    {
        "Name of Bias": "Survivorship Bias",
        "Category of Bias": "Meaning",
        "Description": "Focusing only on the successful examples in a group and overlooking those that failed."
    },
    {
        "Name of Bias": "Subadditivity Effect",
        "Category of Bias": "Meaning",
        "Description": "Judging the probability of a whole event as less than the sum of its individual parts."
    },
    {
        "Name of Bias": "Denomination Effect",
        "Category of Bias": "Meaning",
        "Description": "We are less likely to spend large bills and more likely to spend equivalent smaller amounts of money."
    },
    {
        "Name of Bias": "Illusion of Transparency",
        "Category of Bias": "Meaning",
        "Description": "Assuming that others can understand our thoughts and feelings, often overestimating how well we communicate."
    },
    {
        "Name of Bias": "Curse of Knowledge",
        "Category of Bias": "Meaning",
        "Description": "Once we know something, we assume it’s obvious to others, leading to difficulties in communication."
    },
    {
        "Name of Bias": "Spotlight Effect",
        "Category of Bias": "Meaning",
        "Description": "We overestimate how much others are paying attention to us."
    },
    {
        "Name of Bias": "Extrinsic Incentives Bias",
        "Category of Bias": "Meaning",
        "Description": "We think others are more motivated by external rewards (like money) than intrinsic rewards (like self-improvement)."
    },
    {
        "Name of Bias": "Illusion of External Agency",
        "Category of Bias": "Meaning",
        "Description": "Believing that good or positive things happen due to outside forces, not personal effort."
    },
    {
        "Name of Bias": "Illusion of Asymmetric Insight",
        "Category of Bias": "Meaning",
        "Description": "We believe we know others better than they know themselves."
    },
    {
        "Name of Bias": "Rosy Retrospection",
        "Category of Bias": "Meaning",
        "Description": "Judging the past more positively than it actually was."
    },
    {
        "Name of Bias": "Hindsight Bias",
        "Category of Bias": "Meaning",
        "Description": "Believing that past events were more predictable than they actually were."
    },
    {
        "Name of Bias": "Outcome Bias",
        "Category of Bias": "Meaning",
        "Description": "Judging a decision based on its outcome rather than on the process used to make the decision."
    },
    {
        "Name of Bias": "Impact Bias / Affective Forecasting",
        "Category of Bias": "Meaning",
        "Description": "We overestimate the emotional impact that certain events will have on us."
    },
    {
        "Name of Bias": "Optimism Bias",
        "Category of Bias": "Meaning",
        "Description": "We believe that negative events are less likely to happen to us than to others."
    },
    {
        "Name of Bias": "Planning Fallacy",
        "Category of Bias": "Meaning",
        "Description": "We underestimate how long tasks will take, believing we can complete them more quickly than is realistic."
    },
    {
        "Name of Bias": "Pro-Innovation Bias",
        "Category of Bias": "Meaning",
        "Description": "We overvalue new ideas and innovations, ignoring their weaknesses and limitations."
    },
    {
        "Name of Bias": "Restraint Bias",
        "Category of Bias": "Meaning",
        "Description": "We overestimate our ability to control impulses or avoid temptations."
    },
    {
        "Name of Bias": "Overconfidence Effect",
        "Category of Bias": "Speed",
        "Description": "We believe we make better decisions than we actually do, especially when self-confidence is high."
    },
    {
        "Name of Bias": "Social Desirability Bias",
        "Category of Bias": "Speed",
        "Description": "We answer questions in ways that make us appear more favorable to others, personally and professionally."
    },
    {
        "Name of Bias": "Third-Person Effect",
        "Category of Bias": "Speed",
        "Description": "We overestimate how much mass communication affects others but underestimate its effect on ourselves."
    },
    {
        "Name of Bias": "False Consensus Effect",
        "Category of Bias": "Speed",
        "Description": "We assume that our own behavior and choices are typical, believing others would act similarly in the same situation."
    },
    {
        "Name of Bias": "Head-Easy Effect",
        "Category of Bias": "Speed",
        "Description": "We assume that a difficult task will succeed more quickly than an easy one."
    },
    {
        "Name of Bias": "Dunning-Kruger Effect",
        "Category of Bias": "Speed",
        "Description": "People with low ability overestimate their own skills, while those with high ability underestimate theirs."
    },
    {
        "Name of Bias": "Egocentric Bias",
        "Category of Bias": "Speed",
        "Description": "We place too much value on our own perspective and opinions, often overestimating their importance."
    },
    {
        "Name of Bias": "Barnum Effect / Forer Effect",
        "Category of Bias": "Speed",
        "Description": "We believe vague or general statements (like horoscopes) are personally meaningful when they apply to a large group."
    },
    {
        "Name of Bias": "Self-Serving Bias",
        "Category of Bias": "Speed",
        "Description": "We attribute success to our own ability and failure to external factors to maintain self-esteem."
    },
    {
        "Name of Bias": "Illusion of Control",
        "Category of Bias": "Speed",
        "Description": "We overestimate our ability to influence events that are largely out of our control."
    },
    {
        "Name of Bias": "Illusory Superiority",
        "Category of Bias": "Speed",
        "Description": "We overestimate our own qualities and abilities in comparison to others."
    },
    {
        "Name of Bias": "Trait Ascription Bias",
        "Category of Bias": "Speed",
        "Description": "We consider ourselves unpredictable in terms of personality, behavior, and mood, but view others as predictable."
    },
    {
        "Name of Bias": "Effort Justification",
        "Category of Bias": "Speed",
        "Description": "We place more value on an outcome if we put significant effort into achieving it."
    },
    {
        "Name of Bias": "Risk Compensation",
        "Category of Bias": "Speed",
        "Description": "We act more cautiously when we perceive risk to be high and become less cautious when we feel secure."
    },
    {
        "Name of Bias": "Hyperbolic Discounting / Instant Gratification",
        "Category of Bias": "Speed",
        "Description": "We prefer immediate rewards and unconsciously discount the value of future rewards."
    },
    {
        "Name of Bias": "Appeal to Novelty",
        "Category of Bias": "Speed",
        "Description": "We overestimate the value of something just because it is new, assuming new is always better."
    },
    {
        "Name of Bias": "Identifiable Victim Effect",
        "Category of Bias": "Speed",
        "Description": "We are more willing to help a single identifiable victim than a large group needing help."
    },
    {
        "Name of Bias": "Zeigarnik Effect",
        "Category of Bias": "Speed",
        "Description": "We remember unfinished tasks better than completed ones."
    },
    {
        "Name of Bias": "Sunk Cost Fallacy",
        "Category of Bias": "Speed",
        "Description": "We continue investing in a decision or action due to previous investments (time, money, effort), even when it no longer makes sense."
    },
    {
        "Name of Bias": "Irrational Escalation",
        "Category of Bias": "Speed",
        "Description": "We continue a course of action despite negative outcomes because it aligns with our past decisions."
    },
    {
        "Name of Bias": "Generation Effect",
        "Category of Bias": "Speed",
        "Description": "We remember things better when we generate them ourselves (e.g., ideas, solutions) rather than reading them."
    },
    {
        "Name of Bias": "Loss Aversion / Disposition Effect",
        "Category of Bias": "Speed",
        "Description": "We fear losses more than we value gains, motivating us to avoid losing something over winning."
    },
    {
        "Name of Bias": "IKEA Effect",
        "Category of Bias": "Speed",
        "Description": "We place more value on things we have helped create or assemble."
    },
    {
        "Name of Bias": "Zero-Risk Bias",
        "Category of Bias": "Speed",
        "Description": "We prefer reducing risk in one small area rather than removing a greater overall risk."
    },
    {
        "Name of Bias": "Processing Difficulty Effect",
        "Category of Bias": "Speed",
        "Description": "We remember information better when it requires more cognitive effort to understand."
    },
    {
        "Name of Bias": "Endowment Effect",
        "Category of Bias": "Speed",
        "Description": "We place more value on items we already own compared to the same items if we didn't possess them."
    },
    {
        "Name of Bias": "Backfire Effect",
        "Category of Bias": "Speed",
        "Description": "When confronted with evidence that contradicts our beliefs, we may double down and believe even more strongly."
    },
    {
        "Name of Bias": "Reverse Psychology / Reactance",
        "Category of Bias": "Speed",
        "Description": "People tend to do the opposite of what they are told to do, especially if they perceive their freedom is being restricted."
    },
    {
        "Name of Bias": "Decoy Effect",
        "Category of Bias": "Speed",
        "Description": "People are more likely to make a certain choice when a third, less desirable option is presented."
    },
    {
        "Name of Bias": "Social Comparison Effect",
        "Category of Bias": "Speed",
        "Description": "We tend to feel negatively toward people if they are perceived as better than us, either mentally or physically."
    },
    {
        "Name of Bias": "Less is Better Effect",
        "Category of Bias": "Speed",
        "Description": "We perceive a smaller, high-quality item as more valuable than a larger, low-quality item, even if the latter is worth more."
    },
    {
        "Name of Bias": "Conjunction Fallacy",
        "Category of Bias": "Speed",
        "Description": "We think that specific conditions (e.g., losing the first set but winning the match) are more likely than general events (e.g., winning the match)."
    },
    {
        "Name of Bias": "Law of Triviality",
        "Category of Bias": "Speed",
        "Description": "We spend more time focusing on trivial matters than on more complex, critical issues."
    },
    {
        "Name of Bias": "Rhyme-as-Reason Effect",
        "Category of Bias": "Speed",
        "Description": "We are more likely to believe something if it is phrased in a rhyme."
    },
    {
        "Name of Bias": "Belief Bias",
        "Category of Bias": "Speed",
        "Description": "We consider arguments stronger if they support our pre-existing beliefs."
    },
    {
        "Name of Bias": "Information Bias",
        "Category of Bias": "Speed",
        "Description": "We believe that more information is better and leads to better decisions, even when it's irrelevant."
    },
    {
        "Name of Bias": "Ambiguity Bias",
        "Category of Bias": "Speed",
        "Description": "We prefer options with known outcomes over those with uncertain probabilities, even if the latter may be more favorable."
    },
    {
        "Name of Bias": "Misattribution of Memory",
        "Category of Bias": "Remembering",
        "Description": "We sometimes do not remember things as they actually happened."
    },
    {
        "Name of Bias": "Source Confusion",
        "Category of Bias": "Remembering",
        "Description": "We remember things differently after hearing others talk about the same event."
    },
    {
        "Name of Bias": "Cryptomnesia",
        "Category of Bias": "Remembering",
        "Description": "We recall an idea as our own when we actually encountered it elsewhere."
    },
    {
        "Name of Bias": "False Memory",
        "Category of Bias": "Remembering",
        "Description": "We remember things that we believe happened but in reality, they did not."
    },
    {
        "Name of Bias": "Suggestibility",
        "Category of Bias": "Remembering",
        "Description": "We are susceptible to suggestions from others and may build false memories based on repeated exposure to suggestions."
    },
    {
        "Name of Bias": "Spacing Effect",
        "Category of Bias": "Remembering",
        "Description": "We remember things better when we study them over time with breaks rather than cramming all at once."
    },
    {
        "Name of Bias": "Fading Affect Bias",
        "Category of Bias": "Remembering",
        "Description": "We forget memories with negative associations faster than positive ones."
    },
    {
        "Name of Bias": "Serial-Position Effect",
        "Category of Bias": "Remembering",
        "Description": "We remember the first and last items in a list better than the middle items."
    },
    {
        "Name of Bias": "Recency Effect",
        "Category of Bias": "Remembering",
        "Description": "We remember things that happened most recently better than earlier events."
    },
    {
        "Name of Bias": "Primacy Effect",
        "Category of Bias": "Remembering",
        "Description": "We remember information presented first better than information presented later."
    },
    {
        "Name of Bias": "Memory Inhibition",
        "Category of Bias": "Remembering",
        "Description": "We don't remember irrelevant information because our brain filters out unimportant details."
    },
    {
        "Name of Bias": "Modality Effect",
        "Category of Bias": "Remembering",
        "Description": "The way knowledge is presented (e.g., visually or verbally) affects how well we remember it. Visual presentations are generally better remembered."
    },
    {
        "Name of Bias": "Duration Neglect",
        "Category of Bias": "Remembering",
        "Description": "The perceived value of an experience is determined by its peak moments and its end, rather than its overall duration."
    },
    {
        "Name of Bias": "Serial Recall Effect",
        "Category of Bias": "Remembering",
        "Description": "We are better at recalling items or events in the order they occurred."
    },
    {
        "Name of Bias": "Misinformation Effect",
        "Category of Bias": "Remembering",
        "Description": "Memories can be influenced by new, misleading information that is presented after the event."
    },
    {
        "Name of Bias": "Peak-End Rule",
        "Category of Bias": "Remembering",
        "Description": "Our evaluation of an experience is heavily influenced by how it felt at its peak and its conclusion, rather than the total experience."
    },
    {
        "Name of Bias": "Levels-of-Processing Effect",
        "Category of Bias": "Remembering",
        "Description": "The more deeply we process information, the better we remember it. Engaging deeply with the material improves retention."
    },
    {
        "Name of Bias": "Absent-mindedness",
        "Category of Bias": "Remembering",
        "Description": "We forget things due to a lack of attention, distraction, or being preoccupied with other tasks."
    },
    {
        "Name of Bias": "Testing Effect",
        "Category of Bias": "Remembering",
        "Description": "We retain information better when we are regularly tested on it rather than just reviewing or studying it."
    },
    {
        "Name of Bias": "Next-in-Line Effect",
        "Category of Bias": "Remembering",
        "Description": "We struggle to remember information presented just before our turn to perform, due to anxiety or distraction."
    },
    {
        "Name of Bias": "Google Effect",
        "Category of Bias": "Remembering",
        "Description": "We are more likely to forget information that we know we can easily look up later."
    },
    {
        "Name of Bias": "Tip of the Tongue Phenomenon",
        "Category of Bias": "Remembering",
        "Description": "The temporary inability to recall a word or piece of information, while feeling that retrieval is imminent."
    }
]


# JSON schema definition to validate the model's response format
biases_schema = {
    "type": "object",
    "properties": {
        bias: {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": f"Indicate the valence or level (0-10) for the '{bias}' bias or heuristic.",
                    "minimum": 0,
                    "maximum": 10
                },
                "reasoning": {
                    "type": "string",
                    "description": f"Provide reasoning or justification for the level of '{bias}'."
                }
            },
            "required": ["level", "reasoning"],
            "additionalProperties": False
        } for bias in list([bias['Name of Bias'] for bias in biases_heuristics])
    },
    "required": list([bias['Name of Bias'] for bias in biases_heuristics]),  # Ensure the schema expects the correct list of biases
    "additionalProperties": False
}

def save_response_outputs(response, pathname, filename='videoLog', selected_columns=['id','object','created','model','usage']):
    # Ensure that only dictionaries are processed
    filtered_data = [list([column, response[column]]) for column in selected_columns]
    # Convert the filtered data into a pandas DataFrame
    event_log_details = pd.DataFrame(filtered_data)
    # Save to CSV
    event_log_details.to_csv(OUTPUT_PATH + pathname.split('/')[-1][:-4] + '_' + filename + '_metadata.csv', index=False)
    # Save to CSV
    pd.DataFrame.from_dict(response['choices']).to_csv(OUTPUT_PATH + pathname.split('/')[-1][:-4] + '_' + filename + '_detailedResponse.csv', index=False)


def generate_prompt(videoLog_df, audioLog_df, biases_heuristics):
    # Create a prompt asking the model to evaluate each bias or heuristic.
    prompt = f"""
    You are provided with the video log and audio log (in JSON format) of an online video/television ad.
    Below are the logs:
    Video Log:
    {videoLog_df.to_json(orient='records', lines=False)}
    Audio Log:
    {audioLog_df.to_json(orient='records', lines=False)}
    For each of the biases or heuristics listed below, please evaluate the presence and valence or level (0-10) of the bias or heuristic evident in the video and audio logs.
    Provide a justification for your assessment of the level of each bias or heuristic.
    The biases and heuristics to evaluate are:
    {', '.join(list([bias['Name of Bias'] for bias in biases_heuristics]))}
    Please provide the response in JSON format.
    """
    return prompt


# evaluate_biases_json_schema(event_log, transcription, biases_heuristics, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06")
def evaluate_biases_json_schema(videoLog_df, audioLog_df, biases_heuristics, api_key, model="gpt-4o-mini"): # model = "gpt-4o-2024-08-06"
    """
    Analyzes the prepared frames with OpenAI's API and returns a structured event log.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Generate the prompt
    full_prompt = generate_prompt(videoLog_df, audioLog_df, biases_heuristics)
    messages = [
        {"role": "system","content": "You are an expert in cognitive biases and heuristics."},
        {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "BiasesHeuristicsExtraction",
                "schema": biases_schema #.model_json_schema()
            }
        },
        "max_tokens": 6000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


##### ------ DEFINE FUNCTIONS - END ------- ####

#EXAMPLES
#response = analyze_frames_with_openai_jsonSchema(prepared_frames, prompt, os.environ['OPENAI_API_KEY'],model = "gpt-4o-2024-08-06")
#responseparsed = parse_response_content(response['choices'][0]['message']['content'])
#response2 = analyze_frames_with_openai_sdk(prepared_frames, prompt, os.environ['OPENAI_API_KEY'],model = "gpt-4o-2024-08-06")
#response2parsed = parse_response_content(response2['choices'][0]['message']['content'])

# Find all ".mp4" files in the input file
allfiles = glob.glob(INPUT_PATH + '*.mp4')

# Prompt for OpenAI API
prompt = """These attached images are frames from an online video/television ad, accompanied by an audio transcript in JSON format.
Generate a comprehensive historical action/activity/event log detailing everything that happens at each timestamp in a structured table.
For each event, also include:
- Scene/background context.
- Any behavioral nudges, devices, or moral suasion used.
- Any hidden messages or 'reading between the lines' moments.
- Any use of humor, if present.
At the end of the table, also provide an overall summary or qualitative description of the ad.
Please provide the response in JSON format."""

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
    # Extract the audio transcription and segments
    transcription = transcribe_audio(video_path, segment_timing=True)
    transcription = extract_segments(transcription, full_report=True)
    # Save to CSV
    transcription.to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + '_audioLog.csv', index=False)
    # Slice the dataframes
    transcription = transcription[["id", "start", "end", "text"]]
    # Generate the event log
    try:
        event_log = generate_event_log_or_storyboard_structuredOutput(video_path, frames, transcription, prompt, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06") # Output columns: ["timestamp", "activity_event_action", "scene_background_context", "behavioural_nudge_device_or_moral_suasion"]
    except:
        print('Video file path failed to extract event log: '+video_path)
        continue
    # Save to CSV
    event_log.to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + '_videoLog.csv', index=False)
    # Biases log
    biases_log = evaluate_biases_json_schema(event_log, transcription, biases_heuristics, os.environ['OPENAI_API_KEY'], model="gpt-4o-2024-08-06")
    biases_log = parse_response_content(json.loads(biases_log['choices'][0]['message']['content']))
    # Save the response details to CSV
    save_response_outputs(biases_log, video_path, filename='videoLog')
    # Save to biases evaluation to CSV
    biases_log.to_csv(OUTPUT_PATH + video_path.split('/')[-1][:-4] + '_biases.csv', index=False)


# ---- STEP 2)
