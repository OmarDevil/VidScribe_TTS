import google.generativeai as genai
from docx import Document
from datetime import datetime

API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"

genai.configure(api_key=API_KEY)

fixed_prompt = """
# اكتب تعليقًا صوتيًا (Voice Over) لفيديو مدته 60 ثانية عن الموضوع التالي.
# يجب أن يكون النص طبيعيًا كما لو أن شخصًا يقرأه بصوته فقط، بدون توقيتات أو إشارات موسيقية أو أي عناوين إضافية.
# ابدأ النص مباشرةً بدون كتابة أي عنوان مثل "Script for YouTube Video".
# يجب أن يكون بسيطًا، سهل الفهم، مباشرًا، وكأنه تعليق صوتي احترافي.
# استخدم جُملاً قصيرة وتجنب التفاصيل غير الضرورية أو الكلمات الزائدة.
"""

topic = "talk about updates in ai field in 2025 in english"

final_prompt = fixed_prompt + "\n\nالموضوع: " + topic

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(final_prompt)

text = response.text.strip()

unwanted_titles = ["Script for YouTube Video", "Voice Over Script"]
for title in unwanted_titles:
    if text.startswith(title):
        text = text[len(title):].strip()

doc = Document()
doc.add_paragraph(text)

filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.docx")

doc.save(filename)

print(f"✅ Voice Over Script saved as {filename}")

