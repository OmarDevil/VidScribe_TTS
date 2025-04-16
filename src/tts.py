import torch
from TTS.api import TTS
import os
import glob


# Init TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def text_to_speech(text: str, output_path: str, language: str, speaker: str) -> None:
    """Convert text to speech and save it as a file."""
    tts_model.tts_to_file(
        text=text,
        file_path=output_path,
        language=language,
        speaker=speaker,
        split_sentences=True
    )
    print(f"[INFO] Audio saved to: {output_path}")


def get_txt_file(file_path: str) -> str:
    """Read text from a file."""
    txt_files = glob.glob(os.path.join(file_path, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {file_path}")
    latest_file = max(txt_files, key=os.path.getctime)
    return latest_file


def main():
    file_path = "../video/scripts/"
    try:
        latest_file = get_txt_file(file_path)
        with open(latest_file, "r", encoding="utf-8") as file:
            text = file.read()
        print(f"Text from file: {text}")
    except FileNotFoundError as e:
        print(e)
        text = "Ich bin eine Testnachricht."
    output_path = "../video/voice_over/test.wav"
    language = input("\nchoose language: \n"
                     "English (en)\n"
                     "Arabic (ar)\n"
                     "Spanish (es)\n"
                     "French (fr)\n"
                     "German (de)\n"
                     "Italian (it)\n"
                     "Portuguese (pt)\n"
                     "Chinese (zh-cn)\n"
                     "Korean (ko)\n"
                     "Japanese (ja)\n"
                     "Hindi (hi)\n"
                     "Polish (pl)\n"
                     "Turkish (tr)\n"
                     "Russian (ru)\n"
                     "Dutch (nl)\n"
                     "Czech (cs)\n"
                     "Hungarian (hu)\n")
    speaker = input('\nchoose speaker: \n'
                    'Claribel Dervla\n'
                    'Daisy Studious\n'
                    'Tammie Ema\n'
                    'Gracie Wise\n'
                    'Alison Dietlinde\n'
                    'Ana Florence\n'
                    'Annmarie Nele\n'
                    'Asya Anara\n'
                    'Brenda Stern\n'
                    'Gitta Nikolina\n'
                    'Henriette Usha\n'
                    'Sofia Hellen\n'
                    'Tammy Grit\n'
                    'Tanja Adelina\n'
                    'Vjollca Johnnie\n'
                    'Andrew Chipper\n'
                    'Badr Odhiambo\n'
                    'Dionisio Schuyler\n'
                    'Royston Min\n'
                    'Viktor Eka\n'
                    'Abrahan Mack\n'
                    'Adde Michal\n'
                    'Baldur Sanjin\n'
                    'Craig Gutsy\n'
                    'Damien Black\n'
                    'Gilberto Mathias\n'
                    'Ilkin Urbano\n'
                    'Kazuhiko Atallah\n'
                    'Ludvig Milivoj\n'
                    'Suad Qasim\n'
                    'Torcull Diarmuid\n'
                    'Viktor Menelaos\n'
                    'Zacharie Aimilios\n'
                    'Nova Hogarth\n'
                    'Maja Ruoho\n'
                    'Uta Obando\n'
                    'Lidiya Szekeres\n'
                    'Chandra MacFarland\n'
                    'Szofi Granger\n'
                    'CamillaHolmström\n'
                    'Lilya Stainthorpe\n'
                    'Zofija Kendrick\n'
                    'Narelle Moon\n'
                    'Barbora MacLean\n'
                    'Alexandra Hisakawa\n'
                    'Alma María\n'
                    'Rosemary Okafor\n'
                    'Ige Behringer\n'
                    'Filip Traverse\n'
                    'Damjan Chapman\n'
                    'Wulf Carlevaro\n'
                    'Aaron Dreschner\n'
                    'Kumar Dahl\n'
                    'Eugenio Mataracı\n'
                    'Ferran Simen\n'
                    'Xavier Hayasaka\n'
                    'Luis Moray\n'
                    'Marcos Rudaski\n')
    get_txt_file(file_path)
    text_to_speech(text, output_path, language, speaker)


if __name__ == "__main__":
    main()
