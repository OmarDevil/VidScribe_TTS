import requests
from docx import Document
import os
from datetime import datetime

CHUNK_SIZE = 1024
API_KEY = "sk_bee9b90f2ce467916125923629218a40f6779b2ae28e46dc"

# مسارات المجلدات
SCRIPTS_FOLDER = "scripts"
VOICE_OVER_FOLDER = "voice_over"

# التأكد من إنشاء مجلد الصوتيات إذا لم يكن موجودًا
os.makedirs(VOICE_OVER_FOLDER, exist_ok=True)

url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": API_KEY
}


def read_docx(file_path):
    """ قراءة محتوى ملف وورد """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ الملف {file_path} غير موجود!")

    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)


# البحث عن أحدث ملف سكريبت داخل مجلد scripts
script_files = [f for f in os.listdir(SCRIPTS_FOLDER) if f.startswith("scripts_") and f.endswith(".docx")]
if not script_files:
    raise FileNotFoundError("❌ لم يتم العثور على أي ملف سكريبت داخل مجلد scripts!")

latest_script = max(script_files, key=lambda f: os.path.getctime(os.path.join(SCRIPTS_FOLDER, f)))
script_path = os.path.join(SCRIPTS_FOLDER, latest_script)

print(f"📄 سيتم تحويل الملف: {script_path}")

text_from_docx = read_docx(script_path)

data = {
    "text": text_from_docx,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
}

try:
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()

    # إنشاء اسم ملف صوتي يحتوي على التاريخ والوقت
    audio_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.mp3")
    audio_path = os.path.join(VOICE_OVER_FOLDER, audio_filename)

    with open(audio_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print(f"✅ الصوت تم إنشاؤه وحُفظ في {audio_path}")

except requests.exceptions.RequestException as e:
    print(f"❌ حدث خطأ أثناء الاتصال بـ API: {e}")
