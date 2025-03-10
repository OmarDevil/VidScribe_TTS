import os
import glob
import json
import cv2
import pytesseract
import torch
from youtube_search import YoutubeSearch
import yt_dlp
from ultralytics import YOLO

# مسار ffmpeg (تأكد من تعديله حسب جهازك)
FFMPEG_PATH = r"C:\ffmpeg-2025-03-06-git-696ea1c223-essentials_build\ffmpeg-2025-03-06-git-696ea1c223-essentials_build\bin\ffmpeg.exe"

# تحميل موديل YOLOv5su (إذا لم يكن موجودًا)
model_path = "yolov5su.pt"
if not os.path.exists(model_path):
    torch.hub.download_url_to_file("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt", model_path)

# تحميل الموديل
yolo_model = YOLO(model_path)

def read_keywords():
    """Reads keywords from the first found 'key_words*.txt' file."""
    keyword_files = glob.glob("keeey_words*.txt")
    if not keyword_files:
        print("No keyword file found!")
        return []
    with open(keyword_files[0], "r", encoding="utf-8") as file:
        keywords = [line.strip() for line in file if line.strip()]
    return keywords

def search_videos(query):
    """Searches YouTube for videos matching the query."""
    results = YoutubeSearch(query, max_results=10).to_json()
    videos = json.loads(results).get("videos", [])
    return [video for video in videos if get_video_duration(video['duration']) <= 60]

def get_video_duration(duration_str):
    """Converts YouTube duration (MM:SS or HH:MM:SS) to seconds."""
    if isinstance(duration_str, int):  # If already an integer, return as is
        return duration_str
    parts = list(map(int, duration_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float('inf')

def download_video(video):
    """Downloads the given video using yt-dlp."""
    video_url = f"https://www.youtube.com{video['url_suffix']}"
    output_dir = "downloaded_videos"
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'ffmpeg_location': FFMPEG_PATH,  # تحديد مسار ffmpeg
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def detect_text_in_video(video_path):
    """Detects text in video frames using OpenCV and Tesseract OCR."""
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

def detect_logo_in_video(video_path):
    """Detects logos or watermarks using YOLOv5su."""
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
    keywords = read_keywords()
    if not keywords:
        print("No keywords found!")
        return
    for keyword in keywords:
        print(f"🔍 Searching for: {keyword}")
        videos = search_videos(keyword)
        if not videos:
            print("No short videos found.")
            continue
        for video in videos:
            print(f"⬇ Downloading: {video['title']} ({video['duration']})")
            download_video(video)
            video_path = os.path.join("downloaded_videos", f"{video['title']}.mp4")
            if detect_text_in_video(video_path) or detect_logo_in_video(video_path):
                print("❌ Video contains text or logos, deleting...")
                os.remove(video_path)
            else:
                print("✅ Video is clean.")

if __name__ == "__main__":
    main()
