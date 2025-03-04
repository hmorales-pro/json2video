from flask import Flask, request, jsonify, send_file
import os
import uuid
import subprocess
import pysubs2
import moviepy.editor as mp
from pydub import AudioSegment
from PIL import Image
from dotenv import load_dotenv
from google.cloud import speech
import stat
import time

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Assurez-vous que la clé API Google est configurée
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Route par défaut pour tester le serveur
@app.route('/')
def index():
    return 'Le serveur Flask fonctionne correctement !'

# Route pour générer la vidéo avec sous-titres animés
@app.route('/generate-video', methods=['POST'])
def generate_video_endpoint():
    print("Fichiers reçus :", request.files.keys())

    audio_file = request.files.get('audio')
    image_files = request.files.getlist('images')
    orientation = request.form.get('orientation', 'landscape')
    default_color = request.form.get('default_color', '&HFFFFFF&')

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

    transcription_data = transcribe_audio_in_chunks(wav_path, chunk_duration_ms=30000)

    ass_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.ass")
    generate_animated_ass_subtitles(transcription_data, os.path.abspath(ass_path), default_color, orientation)

    time.sleep(1)

    if not os.path.exists(ass_path):
        print(f"Erreur : Le fichier ASS {ass_path} n'a pas été créé.")
        print("Contenu du répertoire outputs :", os.listdir(OUTPUT_FOLDER))
        return jsonify({'error': 'Subtitle file not created'}), 500

    try:
        with open(ass_path, 'r', encoding='utf-8') as f:
            print("Le fichier ASS est lisible :")
            print(f.read())
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier ASS : {e}")

    ass_stat = os.stat(ass_path)
    print(f"Permissions du fichier ASS : {oct(ass_stat.st_mode)}")

    os.chown(ass_path, 33, 33)
    os.chmod(ass_path, 0o777)

    ffmpeg_test = subprocess.run([
        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=s=1280x720:d=1', '-vf', f'subtitles={os.path.abspath(ass_path)}', '-f', 'null', '-'
    ], capture_output=True, text=True)

    if ffmpeg_test.returncode != 0:
        print("Erreur FFmpeg lors de la lecture du fichier ASS :")
        print(ffmpeg_test.stderr)
        return jsonify({'error': 'FFmpeg cannot read subtitle file'}), 500

    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
    generate_video_with_subtitles(image_paths, wav_path, ass_path, output_path, orientation)

    if not os.path.exists(output_path):
        print(f"Erreur : La vidéo {output_path} n'a pas été créée.")
        return jsonify({'error': 'Video file not created'}), 500

    return send_file(output_path, as_attachment=True)

def generate_animated_ass_subtitles(transcription_data, ass_path, default_color="&HFFFFFF&", orientation="landscape"):
    subs = pysubs2.SSAFile()
    subs.info['Title'] = "Sous-titres Animés"
    style = pysubs2.SSAStyle()
    style.fontname = "Arial"
    style.fontsize = 50

    if default_color.startswith('&H') and default_color.endswith('&'):
        hex_color = default_color[2:-1]
        b, g, r = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        style.primarycolor = pysubs2.Color(r, g, b, 0)
    else:
        r, g, b, a = map(int, default_color.split(','))
        style.primarycolor = pysubs2.Color(r, g, b, a)

    style.backcolor = pysubs2.Color(0, 0, 0, 0)
    style.outline = 1
    style.shadow = 1
    style.alignment = 2 if orientation == "landscape" else 6
    subs.styles['Default'] = style

    for word_info in transcription_data:
        start = word_info['start']
        end = word_info['end']
        text = word_info['word']

        event = pysubs2.SSAEvent()
        event.start = int(start * 1000)
        event.end = int(end * 1000)
        event.text = f"{{\\move(10,500,600,500)}}{text}"
        subs.events.append(event)

    subs.save(ass_path)
    print(f"Fichier ASS généré : {ass_path}")

    if not os.path.exists(ass_path):
        print(f"Erreur : Le fichier ASS {ass_path} n'a pas été créé.")

    print("Contenu du répertoire outputs :", os.listdir(OUTPUT_FOLDER))

# Fonction pour redimensionner les images et vérifier leur validité
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

# Fonction pour générer la vidéo avec sous-titres et audio
def generate_video_with_subtitles(images, audio_path, ass_path, output_path, orientation='landscape'):
    resized_images = resize_and_validate_images(images, orientation)
    clip = mp.ImageSequenceClip(resized_images, fps=1)
    audio = mp.AudioFileClip(audio_path)
    clip = clip.set_duration(audio.duration)
    clip = clip.set_audio(audio)
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    clip.write_videofile(temp_video_path, codec='libx264', fps=24, audio_codec='aac')

    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_video_path, '-vf', f'subtitles="{os.path.abspath(ass_path)}":force_style=FontSize=50', '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k', output_path
    ], capture_output=True, text=True)

    print("FFmpeg subtitle embedding stdout:", result.stdout)
    print("FFmpeg subtitle embedding stderr:", result.stderr)

    os.remove(temp_video_path)

# Fonction pour convertir l'audio en format WAV compatible
def convert_audio_to_wav(audio_path, output_path):
    result = subprocess.run([
        'ffmpeg', '-y', '-i', audio_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', output_path
    ], capture_output=True, text=True)
    print("FFmpeg audio conversion stdout:", result.stdout)
    print("FFmpeg audio conversion stderr:", result.stderr)

def generate_animated_ass_subtitles(transcription_data, ass_path, default_color="&HFFFFFF&", orientation="landscape"):
    subs = pysubs2.SSAFile()
    subs.info['Title'] = "Sous-titres Animés"
    style = pysubs2.SSAStyle()
    style.fontname = "Arial"
    style.fontsize = 50

    if default_color.startswith('&H') and default_color.endswith('&'):
        hex_color = default_color[2:-1]
        b, g, r = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        style.primarycolor = pysubs2.Color(r, g, b, 0)
    else:
        r, g, b, a = map(int, default_color.split(','))
        style.primarycolor = pysubs2.Color(r, g, b, a)

    style.backcolor = pysubs2.Color(0, 0, 0, 0)
    style.outline = 1
    style.shadow = 1
    style.alignment = 2 if orientation == "landscape" else 6
    subs.styles['Default'] = style

    for word_info in transcription_data:
        start = word_info['start']
        end = word_info['end']
        text = word_info['word']

        event = pysubs2.SSAEvent()
        event.start = int(start * 1000)
        event.end = int(end * 1000)
        event.text = f"{{\\move(10,500,600,500)}}{text}"
        subs.events.append(event)

    subs.save(ass_path)
    print(f"Fichier ASS généré : {ass_path}")

    if not os.path.exists(ass_path):
        print(f"Erreur : Le fichier ASS {ass_path} n'a pas été créé.")

    print("Contenu du répertoire outputs :", os.listdir(OUTPUT_FOLDER))

# Fonction pour transcrire l'audio en sous-titres via Google Cloud Speech-to-Text
def transcribe_audio_in_chunks(audio_path, chunk_duration_ms=30000):
    audio_segment = AudioSegment.from_wav(audio_path)
    total_duration_ms = len(audio_segment)
    transcription_data = []
    client = speech.SpeechClient()
    for start_ms in range(0, total_duration_ms, chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk = audio_segment[start_ms:end_ms]
        chunk_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
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

# Fonction pour générer des sous-titres animés au format ASS
def transcribe_audio_in_chunks(audio_path, chunk_duration_ms=30000):
    audio_segment = AudioSegment.from_wav(audio_path)
    total_duration_ms = len(audio_segment)
    transcription_data = []
    client = speech.SpeechClient()
    for start_ms in range(0, total_duration_ms, chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk = audio_segment[start_ms:end_ms]
        chunk_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
