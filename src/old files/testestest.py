from TTS.api import TTS
import torch

# تحميل موديل اللغة اليابانية
model_name = "tts_models/ja/kokoro/tacotron2-DDC"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name).to(device)

# النص الياباني
text = "こんにちは。私はAIです。よろしくお願いします。"

# حفظ الصوت في ملف
tts.tts_to_file(text=text, file_path="output.wav")
print("✅ الصوت تم توليده في output.wav")

# تحميل الموديل
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# طباعة اللغات المدعومة
print("Supported languages:")
print(tts.languages)

from TTS.utils.manage import ModelManager

# إنشاء مدير النماذج
manager = ModelManager()

# جلب كل النماذج المتاحة
all_models = manager.list_models()

# طباعة كل موديل بالاسم ونوعه
for model in all_models:
    print(f"✅ Name: {model['name']}  |  Type: {model['model_type']}")
