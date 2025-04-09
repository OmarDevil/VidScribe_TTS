from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import os
import re
import google.generativeai as genai

# Constants
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
PEXELS_API_KEY = "2aoHx5GeCMB7lOZvplRpapSUQaKFXnRrc3iEP7I4NgimtBUDIybg5GzM"

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Define paths
VIDEO_FOLDER = os.path.join("..", "video")
DOWNLOADED_VIDEOS_FOLDER = os.path.join(VIDEO_FOLDER, "downloaded_videos")
KEYWORDS_FOLDER = os.path.join(VIDEO_FOLDER, "key_words")
SCRIPTS_FOLDER = os.path.join(VIDEO_FOLDER, "scripts")
DOWNLOADED_LINKS_FILE = os.path.join(DOWNLOADED_VIDEOS_FOLDER, "downloaded_links.txt")

# Create folders if they don't exist
os.makedirs(DOWNLOADED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(KEYWORDS_FOLDER, exist_ok=True)
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)


def generate_voice_over_script(topic: str, lang: str = "en") -> str:
    """Generate voice-over script using Gemini API"""
    fixed_prompt = """
    Write a 60-second voice-over script for a video on the following topic.
    The script should be natural, as if someone is reading it aloud.
    Start the script directly without any title.
    Keep it simple and professional.
    """
    final_prompt = fixed_prompt + "\n\nTopic: " + topic + " in " + lang
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(final_prompt)
    return response.text.strip()


def save_script_to_txt(text: str, filename: str) -> None:
    """Save script to text file"""
    file_path = os.path.join(SCRIPTS_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Script saved as {file_path}")


def extract_keywords(text: str, main_topic: str) -> List[str]:
    """Extract 7 clean keywords from the script"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
Extract exactly 7 clean keywords from the following text.
Each keyword should include the word "{main_topic}".
Do NOT include any explanation or intro, just return one keyword per line, with no numbering or extra characters.

Text:
{text}
"""
    response = model.generate_content(prompt)

    # نظف النتيجة: إزالة الترقيم لو موجود
    keywords = [
        re.sub(r'^\d+[\).]?\s*', '', line.strip())
        for line in response.text.strip().split("\n") if line.strip()
    ]
    return keywords[:7]


def save_keywords(keywords: List[str]) -> str:
    """Save keywords to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"key_words_{timestamp}.txt"
    file_path = os.path.join(KEYWORDS_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(keywords))
    return file_path


def search_pexels_videos(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Pexels for videos"""
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={max_results}"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)
    return response.json().get("videos", [])


def sanitize_filename(name: str) -> str:
    """Remove unwanted characters for safe filenames"""
    return re.sub(r'[\\/*?:"<>|]', "", name.replace(" ", "_")).strip()


def is_already_downloaded(video_url: str) -> bool:
    """Check if video was already downloaded"""
    if os.path.exists(DOWNLOADED_LINKS_FILE):
        with open(DOWNLOADED_LINKS_FILE, "r", encoding="utf-8") as f:
            return video_url in f.read()
    return False


def mark_as_downloaded(video_url: str) -> None:
    """Mark video as downloaded"""
    with open(DOWNLOADED_LINKS_FILE, "a", encoding="utf-8") as f:
        f.write(video_url + "\n")


def download_video(video: Dict[str, Any], output_dir: str = DOWNLOADED_VIDEOS_FOLDER) -> Optional[str]:
    """Download video from Pexels using direct MP4 link"""
    os.makedirs(output_dir, exist_ok=True)

    video_url = video["video_files"][0]["link"]
    if is_already_downloaded(video_url):
        print(f"⚠️ Skipping (already downloaded): {video_url}")
        return None

    raw_title = video.get("alt", f"pexels_video_{datetime.now().strftime('%H%M%S')}")
    video_title = sanitize_filename(raw_title)
    file_path = os.path.join(output_dir, f"{video_title}.mp4")

    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        mark_as_downloaded(video_url)
        print(f"✅ Downloaded: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Download failed for {video_title}: {e}")
        return None


def main():
    topic = input("📌 Enter your script topic: ")

    # Generate script
    script = generate_voice_over_script(topic)
    save_script_to_txt(script, f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Extract keywords
    keywords = extract_keywords(script, topic)
    keywords_file = save_keywords(keywords)

    # Download videos
    for keyword in keywords[:3]:  # Use first 3 keywords
        videos = search_pexels_videos(keyword)
        for video in videos[:2]:  # Download 2 videos per keyword
            download_video(video)


if __name__ == "__main__":
    main()
