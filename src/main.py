from typing import List, Optional, Dict, Any
from docx import Document
from datetime import datetime
import requests
import os
import glob
import json
import cv2
import pytesseract
import torch
from youtube_search import YoutubeSearch
import yt_dlp
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import google.generativeai as genai
from gtts import gTTS
import time
from tqdm import tqdm

# Constants
FFMPEG_PATH = r"C:\ffmpeg-2025-03-06-git-696ea1c223-essentials_build\ffmpeg-2025-03-06-git-696ea1c223-essentials_build\bin\ffmpeg.exe"
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
ELEVENLABS_API_KEY = "sk_9cb8fc1fa8d204870d890050a10f6f5e3fc144e1a6b783fd"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)


def generate_voice_over_script(topic: str, lang: str = "en") -> str:
    """
    Generate a voice-over script using Gemini API.
    """
    fixed_prompt = """
    Write a 60-second voice-over script for a video on the following topic.
    The script should be natural, as if someone is reading it aloud, without timings, musical cues, or additional titles.
    Start the script directly without any title like "Script for YouTube Video".
    Keep it simple, easy to understand, direct, and professional.
    Use short sentences and avoid unnecessary details or extra words.
    """
    final_prompt = fixed_prompt + "\n\nTopic: " + topic + " in " + lang
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(final_prompt)
    text = response.text.strip()

    # Remove unwanted titles
    unwanted_titles = ["Script for YouTube Video", "Voice Over Script"]
    for title in unwanted_titles:
        if text.startswith(title):
            text = text[len(title):].strip()

    return text


def save_script_to_docx(text: str, filename: str) -> None:
    """
    Save the generated script to a Word document.
    """
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    print(f"✅ Voice Over Script saved as {filename}")


def extract_key_sentences(text: str) -> List[str]:
    """
    Extract key sentences from the script using Gemini API.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract key sentences from the following script and return them in English:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.split("\n") if response.text else []


def save_keywords(key_sentences: List[str]) -> str:
    """
    Save the extracted key sentences to a text file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"key_words_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for sentence in key_sentences:
            f.write(sentence + "\n")
    print(f"✅ Key sentences saved in {filename}")
    return filename


def search_videos(query: str) -> List[Dict[str, Any]]:
    """
    Search YouTube for videos matching the query.
    """
    results = YoutubeSearch(query, max_results=10).to_json()
    videos = json.loads(results).get("videos", [])
    return [video for video in videos if get_video_duration(video['duration']) <= 60]


def get_video_duration(duration_str: str) -> int:
    """
    Convert YouTube duration (MM:SS or HH:MM:SS) to seconds.
    """
    if isinstance(duration_str, int):
        return duration_str
    parts = list(map(int, duration_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float('inf')


def download_video(video: Dict[str, Any], output_dir: str = "downloaded_videos") -> str:
    """
    Download the given video using yt-dlp.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_url = f"https://www.youtube.com{video['url_suffix']}"
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'ffmpeg_location': FFMPEG_PATH,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return os.path.join(output_dir, f"{video['title']}.mp4")


def detect_text_in_video(video_path: str) -> bool:
    """
    Detect text in video frames using OpenCV and Tesseract OCR.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            print("❌ Text detected in video!")
            cap.release()
            return True
    cap.release()
    return False


def detect_logo_in_video(video_path: str) -> bool:
    """
    Detect logos or watermarks using YOLOv5su.
    """
    model_path = "yolov5su.pt"
    yolo_model = YOLO("yolov5su.pt")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                class_name = yolo_model.names[int(box.cls[0])]
                if class_name in ["logo", "watermark", "text"]:
                    print(f"❌ Logo/Watermark detected: {class_name}")
                    cap.release()
                    return True
    cap.release()
    return False


def convert_text_to_speech(text: str, output_file: str) -> None:
    """
    Convert text to speech using ElevenLabs API.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"✅ Audio saved as {output_file}")


def main():
    # Step 1: Generate Voice Over Script
    print("🚀 Step 1: Generating Voice Over Script...")
    topic = input("Enter your script topic: ")
    lang = input("Choose your language: ")
    with tqdm(total=1, desc="Generating Script") as pbar:
        script_text = generate_voice_over_script(topic, lang)
        pbar.update(1)
    script_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.docx")
    save_script_to_docx(script_text, script_filename)

    # Step 2: Extract Key Sentences
    print("🚀 Step 2: Extracting Key Sentences...")
    with tqdm(total=1, desc="Extracting Key Sentences") as pbar:
        key_sentences = extract_key_sentences(script_text)
        pbar.update(1)
    keywords_filename = save_keywords(key_sentences)

    # Step 3: Search and Download Videos
    print("🚀 Step 3: Searching and Downloading Videos...")
    keywords = open(keywords_filename, "r", encoding="utf-8").read().splitlines()
    for keyword in tqdm(keywords, desc="Processing Keywords"):
        print(f"🔍 Searching for: {keyword}")
        videos = search_videos(keyword)
        if not videos:
            print("No short videos found.")
            continue
        for video in tqdm(videos, desc="Downloading Videos"):
            print(f"⬇ Downloading: {video['title']} ({video['duration']})")
            video_path = download_video(video)
            if detect_text_in_video(video_path) or detect_logo_in_video(video_path):
                print("❌ Video contains text or logos, deleting...")
                os.remove(video_path)
            else:
                print("✅ Video is clean.")

    # Step 4: Convert Script to Speech
    print("🚀 Step 4: Converting Script to Speech...")
    with tqdm(total=1, desc="Converting to Speech") as pbar:
        audio_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.mp3")
        convert_text_to_speech(script_text, audio_filename)
        pbar.update(1)


if __name__ == "__main__":
    main()