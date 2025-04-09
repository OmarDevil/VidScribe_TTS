from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig  # Import the required config class

torch.serialization.add_safe_globals([XttsConfig])

with open("voice_over_20250319_103120.txt", "r", encoding="utf-8") as file:
    text = file.read().strip()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.tts_to_file(
    text=text,
    file_path="test.wav",
    language="ar",
    speaker="Suad Qasim",
    split_sentences=True
)