"""
Génération de vidéos avec sous-titres (pipeline Python / Google Speech).

Endpoints :
    POST /generate-video   (auth X-Api-Key)
    GET  /outputs/<file>   (auth X-Api-Key)
    GET  /monitor          (Basic auth)
    GET  /logs             (Basic auth)
"""
import hmac
import os
import re
import subprocess
import tempfile
import uuid
from functools import wraps

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)
from google.cloud import speech
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

import config  # nos paramètres applicatifs (chargent .env)

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Multer-équivalent : limiter la taille des uploads (anti-DoS).
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------
# Auth API key — comparaison constant-time
# ---------------------------------------------------------------
@app.before_request
def check_api_key():
    # Routes publiques (l'auth Basic dédiée s'en charge le cas échéant).
    if request.endpoint in {"index", "monitor", "get_logs"}:
        return None

    provided_key = request.headers.get("X-Api-Key") or request.args.get("api_key")
    if not provided_key or not hmac.compare_digest(provided_key, config.SECRET_API_KEY):
        return jsonify({"error": "Invalid or missing API key"}), 401
    return None


# ---------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------
def read_image(image_path, target_size=None):
    """Lit une image en BGR (OpenCV) avec fallback Pillow pour GIF/exotiques."""
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[read_image] Échec sur {image_path}: {e}")
            return None
    if target_size is not None:
        try:
            img = cv2.resize(img, target_size)
        except Exception as e:
            print(f"[read_image] Erreur resize {image_path}: {e}")
            return None
    return img


def parse_time(time_str):
    hours, minutes, seconds_millis = time_str.split(":")
    seconds, millis = seconds_millis.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def read_srt(srt_path):
    """Parser SRT minimaliste mais robuste aux lignes vides multiples."""
    subtitles = []
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current = {"start_time": None, "end_time": None, "text": ""}
    for line in lines:
        line = line.strip()
        if line.isdigit():
            if current["start_time"] is not None:
                subtitles.append(
                    (current["start_time"], current["end_time"], current["text"].strip())
                )
            current = {"start_time": None, "end_time": None, "text": ""}
        elif "-->" in line:
            start, end = line.split("-->")
            current["start_time"] = parse_time(start.strip())
            current["end_time"] = parse_time(end.strip())
        elif line:
            current["text"] += " " + line
    if current["start_time"] is not None:
        subtitles.append(
            (current["start_time"], current["end_time"], current["text"].strip())
        )
    return subtitles


def convert_audio_to_wav(audio_path, output_path):
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100",
            output_path,
        ],
        capture_output=True, text=True, check=False,
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"WAV non créé : {output_path}\nFFmpeg stderr: {result.stderr}"
        )


def is_valid_audio_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def is_valid_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ---------------------------------------------------------------
# Transcription Google Speech
# ---------------------------------------------------------------
def transcribe_audio_in_chunks(audio_path, chunk_duration_ms=30000, language_code="fr-FR"):
    audio_segment = AudioSegment.from_wav(audio_path)
    total_duration_ms = len(audio_segment)
    transcription_data = []
    client = speech.SpeechClient()

    for start_ms in range(0, total_duration_ms, chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk = audio_segment[start_ms:end_ms]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
            chunk_path = chunk_file.name
            chunk.export(chunk_path, format="wav")
        try:
            with open(chunk_path, "rb") as audio_file:
                audio_content = audio_file.read()
            audio = speech.RecognitionAudio(content=audio_content)
            recognition_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code=language_code,
                enable_word_time_offsets=True,
            )
            response = client.recognize(config=recognition_config, audio=audio)
            for result in response.results:
                for word_info in result.alternatives[0].words:
                    transcription_data.append(
                        {
                            "word": word_info.word,
                            "start": word_info.start_time.total_seconds() + start_ms / 1000,
                            "end": word_info.end_time.total_seconds() + start_ms / 1000,
                        }
                    )
        except Exception as e:
            print(f"[transcribe_audio_in_chunks] segment KO : {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    return transcription_data


def transcribe_audio_long(audio_path, language_code="fr-FR"):
    client = speech.SpeechClient()
    destination_blob_name = f"audio/{os.path.basename(audio_path)}"
    gcs_uri = upload_file_to_gcs(audio_path, destination_blob_name)

    audio = speech.RecognitionAudio(uri=gcs_uri)
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code=language_code,
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=recognition_config, audio=audio)
    response = operation.result(timeout=600)

    transcription_data = []
    for result in response.results:
        for word_info in result.alternatives[0].words:
            transcription_data.append(
                {
                    "word": word_info.word,
                    "start": word_info.start_time.total_seconds(),
                    "end": word_info.end_time.total_seconds(),
                }
            )
    return transcription_data


def transcribe_audio(audio_path, chunk_duration_ms=30000, threshold_sec=60, language_code="fr-FR"):
    audio_segment = AudioSegment.from_wav(audio_path)
    duration_sec = len(audio_segment) / 1000.0
    if duration_sec <= threshold_sec:
        return transcribe_audio_in_chunks(audio_path, chunk_duration_ms, language_code)
    return transcribe_audio_long(audio_path, language_code)


# ---------------------------------------------------------------
# Génération SRT
# ---------------------------------------------------------------
def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def generate_srt_subtitles(transcription_data, srt_path):
    phrases = []
    current_phrase = []
    current_start = None
    for word_info in transcription_data:
        # Bugfix : on testait `if not current_start` ; 0.0 est falsy → KO.
        if current_start is None:
            current_start = word_info["start"]
        current_phrase.append(word_info["word"])
        if len(current_phrase) >= 5 or word_info.get("pause", False):
            phrases.append(
                {
                    "start": current_start,
                    "end": word_info["end"],
                    "text": " ".join(current_phrase),
                }
            )
            current_phrase = []
            current_start = None
    if current_phrase:
        phrases.append(
            {
                "start": current_start,
                "end": transcription_data[-1]["end"],
                "text": " ".join(current_phrase),
            }
        )
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, phrase in enumerate(phrases, start=1):
            srt_file.write(f"{i}\n")
            srt_file.write(
                f"{_format_timestamp(phrase['start'])} --> {_format_timestamp(phrase['end'])}\n"
            )
            srt_file.write(f"{phrase['text']}\n\n")


def upload_file_to_gcs(source_file_name, destination_blob_name):
    from google.cloud import storage

    bucket_name = config.GCS_BUCKET
    if not bucket_name:
        raise ValueError("GCS_BUCKET non défini (cf. .env)")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"


# ---------------------------------------------------------------
# Wrap texte
# ---------------------------------------------------------------
def wrap_text_by_width(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


# ---------------------------------------------------------------
# Fusion FFmpeg
# ---------------------------------------------------------------
def merge_audio_and_video(video_path, audio_path, output_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            output_path,
        ],
        check=True,
    )


# ---------------------------------------------------------------
# Génération vidéo
# ---------------------------------------------------------------
def generate_video_with_subtitles_opencv(
    image_paths, audio_path, srt_path, output_path,
    fps=24, font_path="DejaVuSans.ttf", font_size=24,
    max_width=1280, max_height=720,
    generate_subtitles=True,
):
    subtitles = read_srt(srt_path) if generate_subtitles else []
    audio = AudioSegment.from_file(audio_path)
    total_duration_sec = len(audio) / 1000.0

    if not image_paths:
        raise ValueError("Aucune image fournie.")
    first_img = read_image(image_paths[0])
    if first_img is None:
        raise ValueError(f"Impossible de lire l'image: {image_paths[0]}")
    orig_height, orig_width, _ = first_img.shape

    width, height = orig_width, orig_height
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        width = int(width * ratio)
        height = int(height * ratio)

    total_frames = int(total_duration_sec * fps)
    segment_duration_sec = total_duration_sec / len(image_paths)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    silent_video_path = output_path.replace(".mp4", "_silent.mp4")
    out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception("VideoWriter non ouvert.")

    global_frame_index = 0
    try:
        for img_path in image_paths:
            img_original = read_image(img_path, target_size=(width, height))
            if img_original is None:
                continue
            frames_for_this_image = int(segment_duration_sec * fps)
            for _ in range(frames_for_this_image):
                current_time = global_frame_index / fps
                current_subtitle = None
                if generate_subtitles:
                    for start, end, text in subtitles:
                        if start <= current_time < end:
                            current_subtitle = text
                            break

                frame = img_original.copy()
                if generate_subtitles and current_subtitle:
                    frame = _draw_subtitle_block(
                        frame, current_subtitle, font_path, font_size, width, height
                    )
                out.write(frame)
                global_frame_index += 1
                if global_frame_index >= total_frames:
                    break
            if global_frame_index >= total_frames:
                break
    finally:
        out.release()

    try:
        merge_audio_and_video(silent_video_path, audio_path, output_path)
    finally:
        if os.path.exists(silent_video_path):
            os.remove(silent_video_path)


def _draw_subtitle_block(frame, text, font_path, font_size, width, height):
    """Dessine un bloc de sous-titre centré (style Instagram) sur la frame."""
    try:
        font_pil = ImageFont.truetype(font_path, font_size)
    except IOError:
        font_pil = ImageFont.load_default()

    dummy_img = Image.new("RGB", (width, height))
    draw_dummy = ImageDraw.Draw(dummy_img)
    max_text_width = int(width * 0.8)
    lines = wrap_text_by_width(draw_dummy, text, font_pil, max_text_width)

    max_line_width = 0
    total_text_height = 0
    line_heights = []
    for ln in lines:
        bbox_ln = draw_dummy.textbbox((0, 0), ln, font=font_pil)
        ln_width = bbox_ln[2] - bbox_ln[0]
        ln_height = bbox_ln[3] - bbox_ln[1]
        max_line_width = max(max_line_width, ln_width)
        line_heights.append(ln_height)
        total_text_height += ln_height

    block_x_center = width // 2
    block_y_center = height // 2
    pad = 20
    left = block_x_center - (max_line_width // 2) - pad
    top = block_y_center - (total_text_height // 2) - pad
    right = block_x_center + (max_line_width // 2) + pad
    bottom = block_y_center + (total_text_height // 2) + pad

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
    current_y = top + pad
    for idx, ln in enumerate(lines):
        bbox_ln = draw.textbbox((0, 0), ln, font=font_pil)
        real_line_width = bbox_ln[2] - bbox_ln[0]
        line_x = block_x_center - (real_line_width // 2)
        draw.text(
            (line_x, current_y),
            ln,
            font=font_pil,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        current_y += line_heights[idx]
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    # Empêche les noms de fichier exotiques (path traversal, caractères de contrôle).
    if not SAFE_FILENAME_RE.match(filename):
        return jsonify({"error": "Nom de fichier invalide"}), 400
    return send_from_directory(
        OUTPUT_FOLDER, filename, as_attachment=True, mimetype="video/mp4"
    )


@app.route("/")
def index():
    return "Le serveur Flask fonctionne correctement !"


@app.route("/generate-video", methods=["POST"])
def generate_video_endpoint():
    audio_file = request.files.get("audio")
    image_files = request.files.getlist("images")

    if not audio_file or not image_files:
        return jsonify({"error": "audio + au moins une image requis"}), 400

    if not is_valid_audio_file(audio_file.filename or ""):
        return jsonify({"error": "Format audio non supporté"}), 400

    cleanup_paths = []
    audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{os.path.basename(audio_file.filename)}")
    audio_file.save(audio_path)
    cleanup_paths.append(audio_path)

    image_paths = []
    for image_file in image_files:
        if not is_valid_image_file(image_file.filename or ""):
            continue
        image_path = os.path.join(
            UPLOAD_FOLDER, f"{uuid.uuid4()}_{os.path.basename(image_file.filename)}"
        )
        image_file.save(image_path)
        image_paths.append(image_path)
        cleanup_paths.append(image_path)

    if not image_paths:
        _safe_unlink(cleanup_paths)
        return jsonify({"error": "Aucune image valide"}), 400

    wav_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
    cleanup_paths.append(wav_path)

    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
    generate_subtitles_param = (
        request.form.get("generate_subtitles", "true").lower() == "true"
    )
    subtitle_lang = request.form.get("subtitle_lang", "fr-FR")

    try:
        convert_audio_to_wav(audio_path, wav_path)

        srt_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.srt")
        cleanup_paths.append(srt_path)
        if generate_subtitles_param:
            transcription_data = transcribe_audio(
                wav_path,
                chunk_duration_ms=30000,
                threshold_sec=60,
                language_code=subtitle_lang,
            )
            generate_srt_subtitles(transcription_data, srt_path)
        else:
            open(srt_path, "w").close()

        generate_video_with_subtitles_opencv(
            image_paths, wav_path, srt_path, output_path,
            fps=24, font_path="DejaVuSans.ttf", font_size=48,
            generate_subtitles=generate_subtitles_param,
        )

        if not os.path.exists(output_path):
            return jsonify({"error": "Video file not created"}), 500

        file_url = request.host_url + "outputs/" + os.path.basename(output_path)
        return jsonify({"url": file_url})

    except Exception as e:
        print(f"[generate_video_endpoint] {e}")
        return jsonify({"error": "Erreur génération vidéo", "details": str(e)}), 500
    finally:
        _safe_unlink(cleanup_paths)


def _safe_unlink(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as e:
            print(f"[cleanup] Impossible de supprimer {p}: {e}")


# ---------------------------------------------------------------
# Monitoring (Basic auth, credentials depuis .env)
# ---------------------------------------------------------------
def check_auth(username, password):
    return hmac.compare_digest(username or "", config.MONITOR_USERNAME) and hmac.compare_digest(
        password or "", config.MONITOR_PASSWORD
    )


def authenticate():
    return Response(
        "Accès non autorisé.",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route("/logs")
@requires_auth
def get_logs():
    """Retourne les dernières lignes de nohup.out (tail au lieu de tout charger)."""
    log_path = "nohup.out"
    max_lines = 500
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()[-max_lines:]
        return jsonify({"logs": "".join(lines)})
    except FileNotFoundError:
        return jsonify({"logs": "(no logs)"})
    except Exception as e:
        return jsonify({"logs": f"Erreur lecture logs: {e}"}), 500


@app.route("/monitor")
@requires_auth
def monitor():
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Monitoring</title>
        <style>
          body { font-family: monospace; background:#f0f0f0; padding:20px; }
          pre { background:#000; color:#0f0; padding:10px; overflow:auto; height:80vh; }
        </style>
      </head>
      <body>
        <h1>Logs (refresh 5s)</h1>
        <pre id="logArea">Chargement...</pre>
        <script>
          async function fetchLogs() {
            try {
              const r = await fetch('/logs', { credentials: 'include' });
              const data = await r.json();
              document.getElementById('logArea').textContent = data.logs;
            } catch (e) {
              document.getElementById('logArea').textContent = "Erreur";
            }
          }
          setInterval(fetchLogs, 5000);
          fetchLogs();
        </script>
      </body>
    </html>
    """
    return render_template_string(html)


if __name__ == "__main__":
    # En prod, lancer avec gunicorn :
    #   gunicorn -w 2 -b 127.0.0.1:5001 app:app
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
