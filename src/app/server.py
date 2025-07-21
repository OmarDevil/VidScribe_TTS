from flask import Flask, request, send_file, jsonify, render_template
from TTS.api import TTS
import torch
import tempfile
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    data = request.json
    text = data.get("text")
    language = data.get("language")
    speaker = data.get("speaker")

    if not text or not language or not speaker:
        return jsonify({"error": "Missing parameters"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        output_path = tmpfile.name
        tts_model.tts_to_file(
            text=text,
            file_path=output_path,
            language=language,
            speaker=speaker,
            split_sentences=True
        )

    return send_file(output_path, mimetype="audio/wav", as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
