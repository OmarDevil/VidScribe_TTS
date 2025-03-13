
import glob
import datetime
import nest_asyncio
import os
from docx import Document
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nest_asyncio.apply()

def find_latest_word_file():
    folder_name = "scripts"
    files = glob.glob(os.path.join(folder_name, "scripts_*.docx"))

    if not files:
        return None

    # فرز الملفات بناءً على التاريخ لاختيار الأحدث
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])


def translate_text(text, target_lang='en'):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)


def extract_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    return keywords


def save_keywords(keywords):
    if not keywords:
        print("\u274c لم يتم العثور على كلمات مفتاحية لاستخراجها.")
        return

    # إنشاء مجلد `voice over` لو مش موجود
    output_folder = "Key Words"
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"key_words_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for keyword in keywords:
            f.write(keyword + "\n")

    print(f"\u2705 تم حفظ الكلمات المفتاحية في {filename}")


def main():
    word_file = find_latest_word_file()

    if word_file:
        print(f"\u2705 تم العثور على الملف: {word_file}")
        text = extract_text_from_docx(word_file)

        if text.strip():
            translated_text = translate_text(text)
            keywords = extract_keywords(translated_text)
            save_keywords(keywords)
        else:
            print("\u274c الملف فارغ أو لا يحتوي على نص قابل للاستخراج.")
    else:
        print("\u274c لم يتم العثور على أي ملف داخل مجلد scripts.")


if __name__ == "__main__":
    main()
