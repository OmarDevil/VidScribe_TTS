import glob
import os
import datetime
from docx import Document
import google.generativeai as genai
from deep_translator import GoogleTranslator
import time

# إعداد مفتاح API الخاص بـ Gemini
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
genai.configure(api_key=GENAI_API_KEY)

def find_word_file():
    """ البحث عن ملف Word يبدأ بـ voice_over """
    files = glob.glob("voice_over*.docx")
    return files[0] if files else None

def extract_text_from_docx(file_path):
    """ استخراج النص من ملف Word """
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def translate_text(text, target_lang='en'):
    """ ترجمة النص إلى الإنجليزية باستخدام Deep Translator """
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"خطأ في الترجمة: {e}")
        return text  # إعادة النص الأصلي في حال الفشل

def get_key_sentences(text):
    """ إرسال النص إلى Gemini API لاستخراج الجمل المفتاحية """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract key sentences from the following script and return them in English:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text.split("\n") if response.text else []
    except Exception as e:
        print(f"خطأ في Gemini API: {e}")
        return []

def save_keywords(key_sentences):
    """ حفظ الجمل المفتاحية في ملف نصي """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keeey_words_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for sentence in key_sentences:
            f.write(sentence + "\n")
    print(f"تم حفظ الجمل المفتاحية في {filename}")

# تنفيذ العمليات
word_file = find_word_file()
if word_file:
    print(f"تم العثور على الملف: {word_file}")
    text = extract_text_from_docx(word_file)
    print("جارٍ الترجمة إلى الإنجليزية...")
    translated_text = translate_text(text)
    time.sleep(2)  # انتظار بسيط لضمان استقرار الترجمة
    print("جارٍ استخراج الجمل المفتاحية...")
    key_sentences = get_key_sentences(translated_text)
    save_keywords(key_sentences)
else:
    print("لم يتم العثور على ملف voice_over.")
