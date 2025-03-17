from flask import Flask, request, jsonify, send_file, send_from_directory, Response, render_template_string
import os
import uuid
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment
import tempfile

from functools import wraps

# On importe la clé depuis un fichier config.py
import config

# Définir des identifiants en dur
USERNAME = "admin"
PASSWORD = "monMotDePasse"

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# ---------------------------------------------------
# Vérification de la clé d'API
# ---------------------------------------------------
@app.before_request
def check_api_key():
    # Autoriser l'accès aux routes 'index' et 'monitor' sans API key.
    if request.endpoint in ['index', 'monitor', 'get_logs']:
        return
    
    provided_key = request.headers.get('X-Api-Key') or request.args.get('api_key')
    if provided_key != config.SECRET_API_KEY:
        return jsonify({"error": "Invalid or missing API key"}), 401

# ---------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------

def read_image(image_path, target_size=None):
    """
    Essaie de lire une image avec cv2.imread.
    Si cela échoue (par exemple pour un GIF), utilise Pillow pour ouvrir l'image, la convertir en RGB,
    puis la convertit en BGR (format OpenCV). Optionnellement, redimensionne l'image à target_size (largeur, hauteur).
    """
    img = cv2.imread(image_path)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f"Image {image_path} chargée via Pillow.")
        except Exception as e:
            print(f"Erreur lors de la lecture de l'image {image_path} avec Pillow : {e}")
            return None
    if target_size is not None:
        try:
            img = cv2.resize(img, target_size)
        except Exception as e:
            print(f"Erreur lors du redimensionnement de l'image {image_path}: {e}")
            return None
    return img

def parse_time(time_str):
    hours, minutes, seconds_millis = time_str.split(':')
    seconds, millis = seconds_millis.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
    return total_seconds

def read_srt(srt_path):
    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    current_sub = {"start_time": None, "end_time": None, "text": ""}
    for line in lines:
        line = line.strip()
        if line.isdigit():
            if current_sub["start_time"] is not None:
                subtitles.append((current_sub["start_time"], current_sub["end_time"], current_sub["text"].strip()))
            current_sub = {"start_time": None, "end_time": None, "text": ""}
        elif '-->' in line:
            start, end = line.split('-->')
            current_sub["start_time"] = parse_time(start.strip())
            current_sub["end_time"] = parse_time(end.strip())
        elif line:
            current_sub["text"] += ' ' + line
    if current_sub["start_time"] is not None:
        subtitles.append((current_sub["start_time"], current_sub["end_time"], current_sub["text"].strip()))
    return subtitles

def resize_and_validate_images(image_paths, orientation='landscape'):
    output_size = (1280, 720) if orientation == 'landscape' else (720, 1280)
    valid_images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img = img.resize(output_size, Image.LANCZOS)
            resized_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_resized.png")
            img.save(resized_path)
            valid_images.append(resized_path)
        except Exception as e:
            print(f"Erreur lors de la validation de l'image {img_path} : {e}")
    return valid_images

def convert_audio_to_wav(audio_path, output_path):
    result = subprocess.run([
        'ffmpeg', '-y', '-i', audio_path, '-acodec', 'pcm_s16le',
        '-ac', '1', '-ar', '44100', output_path
    ], capture_output=True, text=True)
    print("FFmpeg audio conversion stdout:", result.stdout)
    print("FFmpeg audio conversion stderr:", result.stderr)

# ---------------------------------------------------
# Transcription – Deux modes selon la durée de l'audio
# ---------------------------------------------------
def transcribe_audio_in_chunks(audio_path, chunk_duration_ms=30000):
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
        with open(chunk_path, "rb") as audio_file:
            audio_content = audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="fr-FR",
            enable_word_time_offsets=True
        )
        try:
            response = client.recognize(config=config, audio=audio)
            for result in response.results:
                for word_info in result.alternatives[0].words:
                    transcription_data.append({
                        "word": word_info.word,
                        "start": word_info.start_time.total_seconds() + start_ms / 1000,
                        "end": word_info.end_time.total_seconds() + start_ms / 1000
                    })
        except Exception as e:
            print(f"Erreur lors de la transcription du segment : {e}")
        os.remove(chunk_path)
    return transcription_data

def transcribe_audio_long(audio_path):
    client = speech.SpeechClient()
    # Upload the audio file to GCS
    destination_blob_name = f"audio/{os.path.basename(audio_path)}"
    gcs_uri = upload_file_to_gcs(audio_path, destination_blob_name)
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="fr-FR",
        enable_word_time_offsets=True
    )
    
    print("Attente de la transcription asynchrone pour audio long...")
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)  # Ajuste le timeout si nécessaire
    
    transcription_data = []
    for result in response.results:
        for word_info in result.alternatives[0].words:
            transcription_data.append({
                "word": word_info.word,
                "start": word_info.start_time.total_seconds(),
                "end": word_info.end_time.total_seconds()
            })
    return transcription_data

def transcribe_audio(audio_path, chunk_duration_ms=30000, threshold_sec=60):
    audio_segment = AudioSegment.from_wav(audio_path)
    duration_sec = len(audio_segment) / 1000.0
    if duration_sec <= threshold_sec:
        print("Utilisation de la transcription synchrone en chunks")
        return transcribe_audio_in_chunks(audio_path, chunk_duration_ms)
    else:
        print("Utilisation de la transcription asynchrone pour audio long")
        return transcribe_audio_long(audio_path)

# ---------------------------------------------------
# Génération des sous-titres (SRT)
# ---------------------------------------------------
def generate_srt_subtitles(transcription_data, srt_path):
    phrases = []
    current_phrase = []
    current_start = None
    for word_info in transcription_data:
        if not current_start:
            current_start = word_info['start']
        current_phrase.append(word_info['word'])
        if len(current_phrase) >= 5 or (word_info.get('pause', False)):
            phrases.append({
                'start': current_start,
                'end': word_info['end'],
                'text': ' '.join(current_phrase)
            })
            current_phrase = []
            current_start = None
    if current_phrase:
        phrases.append({
            'start': current_start,
            'end': transcription_data[-1]['end'],
            'text': ' '.join(current_phrase)
        })
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, phrase in enumerate(phrases, start=1):
            def format_timestamp(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millisecs = int((seconds - int(seconds)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
            srt_file.write(f"{i}\n")
            srt_file.write(f"{format_timestamp(phrase['start'])} --> {format_timestamp(phrase['end'])}\n")
            srt_file.write(f"{phrase['text']}\n\n")
    print(f"Fichier SRT généré : {srt_path}")

def upload_file_to_gcs(source_file_name, destination_blob_name):
    """
    Uploads a file to a Google Cloud Storage bucket and returns the GCS URI.
    """
    from google.cloud import storage
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise ValueError("GCS_BUCKET environment variable is not set")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    # Optionally, you can make the blob public:
    # blob.make_public()
    return f"gs://{bucket_name}/{destination_blob_name}"


# ---------------------------------------------------
# Fonctions pour dessiner le texte avec fond "Instagram"
# ---------------------------------------------------
def draw_text_pil(frame, text, x, y,
                  font_path="DejaVuSans.ttf", font_size=24,
                  text_color=(255, 255, 255),
                  stroke_color=(0, 0, 0), stroke_width=2,
                  bg_color=None, padding=5):
    """
    Dessine du texte (UNE SEULE LIGNE) sur `frame` (OpenCV) via Pillow.
    Si bg_color est défini, un rectangle de fond est dessiné derrière cette ligne.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Erreur: police '{font_path}' non trouvée. Utilisation de la police par défaut.")
        font = ImageFont.load_default()
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font, stroke_width=stroke_width)
        bg_x0 = bbox[0] - padding
        bg_y0 = bbox[1] - padding
        bg_x1 = bbox[2] + padding
        bg_y1 = bbox[3] + padding
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=bg_color)
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=stroke_width, stroke_fill=stroke_color)
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_bgr

# ---------------------------------------------------
# Fusion audio + vidéo avec ffmpeg
# ---------------------------------------------------
def merge_audio_and_video(video_path, audio_path, output_path):
    import subprocess
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    subprocess.run(cmd, check=True)

# ---------------------------------------------------
# Fonction de découpe en lignes (wrap text)
# ---------------------------------------------------
def wrap_text_by_width(draw, text, font, max_width):
    """
    Découpe `text` en plusieurs lignes pour qu'aucune ne dépasse `max_width`.
    Retourne une liste de lignes.
    """
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines

# ---------------------------------------------------
# Génération vidéo avec sous-titres "Instagram"
# ---------------------------------------------------
def generate_video_with_subtitles_opencv(image_paths, audio_path, srt_path, output_path,
                                         fps=24, font_path="DejaVuSans.ttf", font_size=24,
                                         max_width=1280, max_height=720):
    # Lecture des sous-titres et de l'audio
    print("Lecture des sous-titres depuis", srt_path)
    subtitles = read_srt(srt_path)
    
    print("Lecture de l'audio depuis", audio_path)
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000.0

    if not image_paths:
        raise ValueError("Aucune image fournie.")
    
    # Lecture de la première image avec read_image pour obtenir les dimensions
    first_img = read_image(image_paths[0])
    if first_img is None:
        raise ValueError("Impossible de lire l'image: " + image_paths[0])
    orig_height, orig_width, _ = first_img.shape

    # Redimensionnement global si nécessaire
    width, height = orig_width, orig_height
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        print(f"Redimensionnement: {width}x{height} -> {new_width}x{new_height}")
        width, height = new_width, new_height
    else:
        print(f"Dimensions utilisées: {width}x{height}")

    total_frames = int(total_duration_sec * fps)
    segment_duration_sec = total_duration_sec / len(image_paths)
    print(f"Durée totale : {total_duration_sec:.2f} s, {total_frames} frames.")

    # Création du VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    silent_video_path = output_path.replace('.mp4', '_silent.mp4')
    out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise Exception("Le VideoWriter n'a pas pu être ouvert.")

    global_frame_index = 0
    num_images = len(image_paths)
    print("Début de l'écriture des frames...")
    try:
        for idx, img_path in enumerate(image_paths):
            print(f"Traitement de l'image {idx+1}/{num_images}: {img_path}")
            # Utiliser read_image pour gérer tous les formats
            img_original = read_image(img_path, target_size=(width, height))
            if img_original is None:
                print("Avertissement : image ignorée -", img_path)
                continue
            frames_for_this_image = int(segment_duration_sec * fps)
            for i in range(frames_for_this_image):
                current_time = global_frame_index / fps
                current_subtitle = None
                for start, end, text in subtitles:
                    if start <= current_time < end:
                        current_subtitle = text
                        break

                frame = img_original.copy()
                if current_subtitle:
                    # Créer une image dummy pour mesurer le texte
                    dummy_img = Image.new("RGB", (width, height))
                    draw_dummy = ImageDraw.Draw(dummy_img)
                    try:
                        font_pil = ImageFont.truetype(font_path, font_size)
                    except IOError:
                        print(f"Erreur: police '{font_path}' non trouvée. Utilisation de la police par défaut.")
                        font_pil = ImageFont.load_default()
                    max_text_width = int(width * 0.8)
                    lines = wrap_text_by_width(draw_dummy, current_subtitle, font_pil, max_text_width)
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
                    block_padding = 20
                    block_left   = block_x_center - (max_line_width // 2) - block_padding
                    block_top    = block_y_center - (total_text_height // 2) - block_padding
                    block_right  = block_x_center + (max_line_width // 2) + block_padding
                    block_bottom = block_y_center + (total_text_height // 2) + block_padding

                    # Convertir la frame en image PIL pour dessiner le bloc de texte
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    # Dessiner le rectangle de fond noir
                    draw.rectangle([block_left, block_top, block_right, block_bottom], fill=(0, 0, 0))
                    current_y = block_top + block_padding
                    for i, ln in enumerate(lines):
                        bbox_ln = draw.textbbox((0, 0), ln, font=font_pil)
                        real_line_width = bbox_ln[2] - bbox_ln[0]
                        line_x = block_x_center - (real_line_width // 2)
                        draw.text(
                            (line_x, current_y),
                            ln,
                            font=font_pil,
                            fill=(255, 255, 255),
                            stroke_width=2,
                            stroke_fill=(0, 0, 0)
                        )
                        current_y += line_heights[i]
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(frame)
                global_frame_index += 1
                if global_frame_index % 500 == 0:
                    print(f"{global_frame_index} frames traitées...")
    except Exception as e:
        print("Erreur lors du traitement des frames :", e)
        out.release()
        raise

    out.release()
    print("Frames écrites. Début de la fusion audio/vidéo...")
    try:
        merge_audio_and_video(silent_video_path, audio_path, output_path)
        print("Fusion audio/vidéo terminée.")
    except Exception as e:
        print("Erreur lors de la fusion audio/vidéo :", e)
        raise

    if os.path.exists(silent_video_path):
        os.remove(silent_video_path)
    print(f"Vidéo finale générée : {output_path}")


# ---------------------------------------------------
# Routes Flask
# ---------------------------------------------------
@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True, mimetype='video/mp4')

@app.route('/')
def index():
    return 'Le serveur Flask fonctionne correctement !'

@app.route('/generate-video', methods=['POST'])
def generate_video_endpoint():
    print("Fichiers reçus :", request.files.keys())
    audio_file = request.files.get('audio')
    image_files = request.files.getlist('images')
    orientation = request.form.get('orientation', 'landscape')
    if not audio_file or not image_files:
        return jsonify({'error': 'Audio file and at least one image file are required'}), 400
    audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{audio_file.filename}")
    audio_file.save(audio_path)
    image_paths = []
    for image_file in image_files:
        if image_file.filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{image_file.filename}")
            image_file.save(image_path)
            image_paths.append(image_path)
        else:
            print(f"Fichier ignoré car non valide : {image_file.filename}")
    wav_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
    convert_audio_to_wav(audio_path, wav_path)
    # Utilise la fonction transcribe_audio qui s'adapte selon la durée
    transcription_data = transcribe_audio(wav_path, chunk_duration_ms=30000, threshold_sec=60)
    srt_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.srt")
    generate_srt_subtitles(transcription_data, srt_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
    generate_video_with_subtitles_opencv(image_paths, wav_path, srt_path, output_path,
                                         fps=24,
                                         font_path="DejaVuSans.ttf",
                                         font_size=48)
    if not os.path.exists(output_path):
        print(f"Erreur : La vidéo {output_path} n'a pas été créée.")
        return jsonify({'error': 'Video file not created'}), 500
    filename = os.path.basename(output_path)
    file_url = request.host_url + 'outputs/' + filename
    return jsonify({'url': file_url})


def check_auth(username, password):
    """Vérifie que le couple utilisateur/mot de passe est correct."""
    return username == USERNAME and password == PASSWORD

def authenticate():
    """Envoie une réponse 401 pour demander l'authentification."""
    return Response(
        'Accès non autorisé.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/logs')
@requires_auth
def get_logs():
    try:
        with open("nohup.out", "r") as f:
            log_content = f.read()
    except Exception as e:
        log_content = "Erreur lors de la lecture des logs: " + str(e)
    return jsonify({"logs": log_content})


@app.route('/monitor')
@requires_auth
def monitor():
    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Monitoring de l'application</title>
        <style>
          body { font-family: monospace; background-color: #f0f0f0; padding: 20px; }
          pre { background-color: #000; color: #0f0; padding: 10px; overflow: auto; height: 80vh; }
        </style>
      </head>
      <body>
        <h1>Logs de l'application (mise à jour toutes les 5 secondes)</h1>
        <pre id="logArea">Chargement...</pre>
        <script>
          async function fetchLogs() {
            try {
              const response = await fetch('/logs');
              const data = await response.json();
              document.getElementById('logArea').textContent = data.logs;
            } catch (error) {
              document.getElementById('logArea').textContent = "Erreur lors du chargement des logs";
            }
          }
          // Actualise toutes les 5 secondes
          setInterval(fetchLogs, 5000);
          fetchLogs();
        </script>
      </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
