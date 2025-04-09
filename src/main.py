from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
import os
import cv2
import pytesseract
import yt_dlp
from ultralytics import YOLO
import google.generativeai as genai
from tqdm import tqdm
import json
import time
from youtube_search import YoutubeSearch
import subprocess
import sys
#
# Constants
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
PEXELS_API_KEY = "LtFO8qCo0QX7i5imZWoNeKcuseBMOLE4e31zcVwdno5FDhubyj3GZpMV"  # Replace with your Pexels API key

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Define paths
VIDEO_FOLDER = os.path.join("..", "video")  # Path to the video folder
DOWNLOADED_VIDEOS_FOLDER = os.path.join(VIDEO_FOLDER, "downloaded_videos")
KEYWORDS_FOLDER = os.path.join(VIDEO_FOLDER, "key_words")
SCRIPTS_FOLDER = os.path.join(VIDEO_FOLDER, "scripts")
VOICE_OVER_FOLDER = os.path.join(VIDEO_FOLDER, "voice_over")

# Create folders if they don't exist
os.makedirs(DOWNLOADED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(KEYWORDS_FOLDER, exist_ok=True)
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)
os.makedirs(VOICE_OVER_FOLDER, exist_ok=True)


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


def save_script_to_txt(text: str, filename: str) -> None:
    """
    Save the generated script to a text file in the scripts folder.
    """
    file_path = os.path.join(SCRIPTS_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Voice Over Script saved as {file_path}")


def extract_keywords(text: str, main_topic: str) -> List[str]:
    """
    Extract important keywords from the script and ensure each keyword contains the main topic only once.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (f"Extract the most important keywords from the following script and return them as a comma-separated "
              f"list in English. Ensure each keyword includes '{main_topic}' only once:\n\n{text}")

    response = model.generate_content(prompt)
    if response.text:
        # Split the text into a list of keywords
        keywords = response.text.strip().split(",")

        # Clean the keywords from extra spaces
        cleaned_keywords = [kw.strip() for kw in keywords]

        # Ensure the main topic is included only once in each keyword
        final_keywords = []
        for kw in cleaned_keywords:
            if main_topic.lower() in kw.lower():
                # Remove duplicates if found
                kw = kw.replace(main_topic, "").strip()
                kw = f"{main_topic} {kw}".strip()  # Ensure the main topic is at the beginning
            else:
                kw = f"{main_topic} {kw}".strip()  # If not present, add it naturally

            final_keywords.append(kw)

        return final_keywords

    return []


def save_keywords(keywords: List[str]) -> str:
    """
    Save the extracted keywords to a text file in the key_words folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"key_words_{timestamp}.txt"
    file_path = os.path.join(KEYWORDS_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        for keyword in keywords:
            f.write(keyword + "\n")
    print(f"✅ Keywords saved in {file_path}")
    return file_path


def search_youtube_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search YouTube for videos matching the query.
    """
    try:
        results = YoutubeSearch(query, max_results=max_results).to_json()
        videos = json.loads(results).get("videos", [])
        return [video for video in videos if get_video_duration(video['duration']) <= 60]
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return []


def search_dailymotion_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Dailymotion for videos matching the query.
    """
    url = "https://api.dailymotion.com/videos"
    params = {
        "search": query,
        "limit": max_results,
        "fields": "id,title,duration,url",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return [video for video in data.get("list", []) if video.get("duration", float('inf')) <= 60]
    return []


def search_pexels_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Pexels for videos matching the query.
    """
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={max_results}"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        videos = response.json().get("videos", [])
        return [video for video in videos if video.get("duration", float('inf')) <= 60]
    return []


def get_video_duration(duration_str: str) -> float:
    """
    Convert video duration (MM:SS or HH:MM:SS) to seconds.
    """
    if isinstance(duration_str, int):
        return duration_str
    parts = list(map(int, duration_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float('inf')


def download_video(video: Dict[str, Any], platform: str = "youtube", output_dir: str = DOWNLOADED_VIDEOS_FOLDER) -> \
Optional[str]:
    """
    Download the given video using yt-dlp.
    """
    os.makedirs(output_dir, exist_ok=True)

    if platform == "youtube":
        video_url = f"https://www.youtube.com{video['url_suffix']}"
    elif platform == "dailymotion":
        video_url = video.get("url")
    elif platform == "pexels":
        video_url = video["video_files"][0]["link"]  # Use the first available video file
    else:
        print(f"❌ Unsupported platform: {platform}")
        return None

    # Check if the video is a live stream
    if is_live_stream(video_url):
        print(f"❌ Skipping live stream: {video['title']}")
        return None

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo[height<=480][ext=mp4]/bestvideo[ext=mp4]',  # 480p quality or best available
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return os.path.join(output_dir, f"{video['title']}.mp4")


def is_live_stream(video_url: str) -> bool:
    """
    Check if the video is a live stream.
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info.get('is_live', False)


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


def main():
    steps = 4  # Total number of main steps
    print("\n🚀 Starting the process...\n")

    # Step 1: Get User Input Before Starting the Progress Bar
    topic = input("\n📌 Enter your script topic: ")

    # Configure the progress bar with custom styling
    progress_bar = tqdm(
        total=steps,
        desc="🔄 Progress",
        colour="green",
        bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} Steps"
    )

    # Step 1: Generate Voice Over Script
    print("\n✍️ Generating Voice Over Script...")
    script_text = generate_voice_over_script(topic)
    script_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.txt")
    save_script_to_txt(script_text, script_filename)
    progress_bar.update(1)

    # Step 2: Extract Key Sentences
    print("\n📑 Extracting Key Sentences...")
    key_sentences = extract_keywords(script_text, topic)  # Pass the main topic to include it in keywords
    keywords_filename = save_keywords(key_sentences)
    progress_bar.update(1)

    # Step 3: Search and Download Videos
    print("\n🎥 Searching and Downloading Videos...")
    keywords = open(keywords_filename, "r", encoding="utf-8").read().splitlines()
    for keyword in keywords:
        print(f"🔍 Searching for videos with title containing: {keyword}")

        # Search on YouTube
        videos = search_youtube_videos(keyword)
        if not videos:
            print(f"⚠ No videos found on YouTube, searching on Dailymotion...")
            videos = search_dailymotion_videos(keyword)  # Search on Dailymotion if no results on YouTube

        if not videos:
            print(f"⚠ No videos found on Dailymotion, searching on Pexels...")
            videos = search_pexels_videos(keyword)  # Search on Pexels if no results on Dailymotion

        if not videos:
            print(f"❌ No videos found on any platform for: {keyword}")
            continue

        for video in videos:
            platform = "youtube" if "url_suffix" in video else "dailymotion" if "url" in video else "pexels"
            print(f"⬇ Downloading from {platform}: {video['title']} ({video.get('duration', 'N/A')})")
            video_path = download_video(video, platform)
            if video_path is None:  # Skip if the video is a live stream
                continue
            if detect_text_in_video(video_path) or detect_logo_in_video(video_path):
                print("❌ Video contains text or logos, deleting...")
                os.remove(video_path)
            else:
                print("✅ Video is clean.")
    progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
    print("\n✅ Process completed successfully!\n")


if __name__ == "__main__":
    main()
