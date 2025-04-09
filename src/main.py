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
from youtube_search import YoutubeSearch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
ELEVENLABS_API_KEY = "sk_9cb8fc1fa8d204870d890050a10f6f5e3fc144e1a6b783fd"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
PEXELS_API_KEY = "2aoHx5GeCMB7lOZvplRpapSUQaKFXnRrc3iEP7I4NgimtBUDIybg5GzM"  # Replace with your Pexels API key

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
    Returns a maximum of 10 keywords.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (f"Extract the most important keywords from the following script and return them as a comma-separated "
              f"list in English. Ensure each keyword includes '{main_topic}' only once. Return a maximum of 10 keywords:\n\n{text}")

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

        # Return only the first 10 keywords
        return final_keywords[:10]

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


def search_pexels_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Pexels for videos matching the query.
    """
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={max_results}"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        videos = response.json().get("videos", [])
        for video in videos:
            if "alt" not in video:  # Add 'alt' key if it doesn't exist
                video["alt"] = f"Pexels Video {video.get('id', 'Unknown')}"
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


def download_video(video: Dict[str, Any], platform: str = "youtube", output_dir: str = DOWNLOADED_VIDEOS_FOLDER) -> Optional[str]:
    """
    Download the given video using yt-dlp.
    """
    os.makedirs(output_dir, exist_ok=True)

    if platform == "youtube":
        video_url = f"https://www.youtube.com{video['url_suffix']}"
        video_title = video.get("title", "Unknown Title")
    elif platform == "pexels":
        video_url = video["video_files"][0]["link"]  # Use the first available video file
        video_title = video.get("alt", "Unknown Title")  # Use 'alt' as the title for Pexels videos
    else:
        print(f"❌ Unsupported platform: {platform}")
        return None

    # Check if the video is a live stream
    if is_live_stream(video_url):
        print(f"❌ Skipping live stream: {video_title}")
        return None

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, f"{video_title}.%(ext)s"),  # Use the video title in the output filename
        'format': 'bestvideo[height<=480][ext=mp4]/best[ext=mp4]/best',  # Fallback to any available format
        'quiet': True,  # Suppress yt-dlp output (including warnings)
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return os.path.join(output_dir, f"{video_title}.mp4")
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Failed to download video: {e}")
        return None


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


def detect_text_in_video_fast(video_path: str, frame_interval: int = 90) -> bool:
    """
    Detect text in video frames using OpenCV and Tesseract OCR (optimized).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (192, 320))  # Reduce frame size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                print("❌ Text detected in video!")
                cap.release()
                return True
        frame_count += 1
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
        frame = cv2.resize(frame, (192, 320))  # Reduce frame size
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
    Convert text to speech using ElevenLabs API and save it in the voice_over folder.
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
    file_path = os.path.join(VOICE_OVER_FOLDER, output_file)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"✅ Audio saved as {file_path}")


def search_and_download_videos_parallel(keyword: str, max_videos_per_platform: int = 2) -> None:
    """
    Search and download videos from YouTube and Pexels for a given keyword in parallel.
    Downloads a maximum of `max_videos_per_platform` videos from each platform.
    """
    platforms = [
        ("youtube", search_youtube_videos),
        ("pexels", search_pexels_videos),
    ]

    with ThreadPoolExecutor() as executor:
        futures = []
        for platform, search_function in platforms:
            print(f"🔍 Searching for videos on {platform} with title containing: {keyword}")
            videos = search_function(keyword, max_results=max_videos_per_platform * 2)

            if not videos:
                print(f"⚠ No videos found on {platform} for: {keyword}")
                continue

            for video in videos[:max_videos_per_platform]:
                futures.append(executor.submit(download_video, video, platform))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            video_path = future.result()
            if video_path:
                if detect_text_in_video_fast(video_path) or detect_logo_in_video(video_path):
                    print("❌ Video contains text or logos, deleting...")
                    os.remove(video_path)
                else:
                    print("✅ Video is clean.")


def loading_animation(duration: int = 3):
    """
    Display a simple loading animation.
    """
    for _ in range(duration):
        for symbol in "⣾⣽⣻⢿⡿⣟⣯⣷":
            print(f"\rLoading {symbol}", end="", flush=True)
            time.sleep(0.1)
    print("\rLoading complete! ✅")


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
    loading_animation(2)  # Simulate loading
    script_text = generate_voice_over_script(topic)
    script_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.txt")
    save_script_to_txt(script_text, script_filename)
    progress_bar.update(1)

    # Step 2: Extract Key Sentences
    print("\n📑 Extracting Key Sentences...")
    loading_animation(2)  # Simulate loading
    key_sentences = extract_keywords(script_text, topic)  # Pass the main topic to include it in keywords
    keywords_filename = save_keywords(key_sentences)
    progress_bar.update(1)

    # Step 3: Search and Download Videos
    print("\n🎥 Searching and Downloading Videos...")
    keywords = open(keywords_filename, "r", encoding="utf-8").read().splitlines()
    for keyword in keywords[:2]:  # Use only the first 2 keywords
        search_and_download_videos_parallel(keyword, max_videos_per_platform=2)  # Download 2 videos per platform
    progress_bar.update(1)

    # Step 4: Convert Script to Speech
    print("\n🔊 Converting Script to Speech...")
    loading_animation(2)  # Simulate loading
    audio_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.mp3")
    convert_text_to_speech(script_text, audio_filename)
    progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
    print("\n✅ Process completed successfully!\n")


if __name__ == "__main__":
    main()