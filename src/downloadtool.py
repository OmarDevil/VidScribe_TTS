import time
import os
import yt_dlp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# إعداد متصفح Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # تشغيل المتصفح في الخلفية
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# تشغيل WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def search_tiktok_videos(keyword, max_results=5):
    search_url = f"https://www.tiktok.com/search?q={keyword}"
    driver.get(search_url)
    time.sleep(5)  # انتظار تحميل الصفحة

    video_links = []
    videos = driver.find_elements(By.XPATH, "//a[contains(@href, '/video/')]")

    for video in videos[:max_results]:
        link = video.get_attribute("href")
        if link and link not in video_links:
            video_links.append(link)

    return video_links


def download_tiktok_video(url):
    ydl_opts = {
        'outtmpl': '%(title)s.%(ext)s',  # تسمية الملف باسم العنوان
        'format': 'best',  # تحميل بأفضل جودة متاحة
        'cookies': 'cookies.txt'  # استخدام ملف الكوكيز لتجاوز الحماية
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"⚠️ تعذر تحميل الفيديو: {url}\nالسبب: {e}")


# البحث عن الملفات النصية داخل مجلد "Key Words"
keywords_dir = "Key Words"
keywords_files = [f for f in os.listdir(keywords_dir) if f.endswith(".txt")]

keywords = []
for file_name in keywords_files:
    file_path = os.path.join(keywords_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            keywords.extend([line.strip() for line in file if line.strip()])
    except FileNotFoundError:
        print(f"تعذر العثور على الملف: {file_name}")

# تنفيذ البحث والتحميل
if keywords:
    for keyword in keywords:
        print(f"جاري البحث عن فيديوهات لكلمة: {keyword}")
        video_urls = search_tiktok_videos(keyword)

        if video_urls:
            print("جاري تحميل الفيديوهات...")
            for url in video_urls:
                download_tiktok_video(url)
            print("تم التحميل بنجاح!")
        else:
            print("لم يتم العثور على فيديوهات.")
else:
    print("لا توجد كلمات مفتاحية في أي ملف داخل المجلد.")

# إغلاق المتصفح
driver.quit()

# تحديث yt-dlp تلقائيًا لضمان التوافق مع TikTok
os.system("yt-dlp -U")