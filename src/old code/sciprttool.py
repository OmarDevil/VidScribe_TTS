import google.generativeai as genai
from docx import Document
from datetime import datetime
import os

API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"

genai.configure(api_key=API_KEY)

fixed_prompt = """
# اكتب تعليقًا صوتيًا (Voice Over) لفيديو مدته 60 ثانية عن الموضوع التالي.
# يجب أن يكون النص طبيعيًا كما لو أن شخصًا يقرأه بصوته فقط، بدون توقيتات أو إشارات موسيقية أو أي عناوين إضافية.
# ابدأ النص مباشرةً بدون كتابة أي عنوان مثل "Script for YouTube Video".
# يجب أن يكون بسيطًا، سهل الفهم، مباشرًا، وكأنه تعليق صوتي احترافي.
# استخدم جُملاً قصيرة وتجنب التفاصيل غير الضرورية أو الكلمات الزائدة.
"""

topic = input('Enter your script :')
lang = input('Choose your language :')
final_prompt = fixed_prompt + "\n\nالموضوع: " + topic + ' in ' + lang

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(final_prompt)

text = response.text.strip()

unwanted_titles = ["Script for YouTube Video", "Voice Over Script"]
for title in unwanted_titles:
    if text.startswith(title):
        text = text[len(title):].strip()

# إنشاء فولدر scripts لو مش موجود
folder_name = "scripts"
os.makedirs(folder_name, exist_ok=True)

# توليد اسم الملف باستخدام الوقت والتاريخ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"scripts_{timestamp}.docx"
file_path = os.path.join(folder_name, file_name)

# حفظ الملف داخل الفولدر
doc = Document()
doc.add_paragraph(text)
doc.save(file_path)

print(f"✅ Script saved as {file_path}")

