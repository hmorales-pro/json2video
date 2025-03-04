from flask import Flask, request, jsonify, send_file
import os
import uuid
import subprocess
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google.cloud import speech
from pydub import AudioSegment
import tempfile

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Assurez-vous que la clé API Google est configurée
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def parse_time(time_str):
    """
    Convertit un timestamp SRT (HH:MM:SS,mmm) en nombre de secondes (float).
    """
    hours, minutes, seconds_millis = time_str.split(':')
    seconds, millis = seconds_millis.split(',')
    total_seconds = (int(hours) * 3600
                     + int(minutes) * 60
                     + int(seconds)
                     + int(millis) / 1000.0)
    return total_seconds

def read_srt(srt_path):
    """
    Lit un fichier SRT et retourne une liste de tuples (start, end, text).
    """
    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_sub = {
        "start_time": None,
        "end_time": None,
        "text": ""
    }

    for line in lines:
        line = line.strip()

        # Numéro de sous-titre
        if line.isdigit():
            # Si on avait un bloc en cours, on l'ajoute à la liste
            if current_sub["start_time"] is not None:
                subtitles.append((
                    current_sub["start_time"],
                    current_sub["end_time"],
                    current_sub["text"].strip()
                ))
            # On réinitialise pour le nouveau bloc
            current_sub = {
                "start_time": None,
                "end_time": None,
                "text": ""
            }

        # Ligne qui contient --> (ex: "00:00:01,000 --> 00:00:03,000")
        elif '-->' in line:
            start, end = line.split('-->')
            current_sub["start_time"] = parse_time(start.strip())
            current_sub["end_time"]   = parse_time(end.strip())

        # Texte du sous-titre
        elif line:
            # On ajoute la ligne de texte (ici en concaténant sur une seule ligne)
            current_sub["text"] += ' ' + line

    # Ajouter le dernier bloc si existant
    if current_sub["start_time"] is not None:
        subtitles.append((
            current_sub["start_time"],
            current_sub["end_time"],
            current_sub["text"].strip()
        ))

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
        'ffmpeg', '-y', '-i', audio_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', output_path
    ], capture_output=True, text=True)
    print("FFmpeg audio conversion stdout:", result.stdout)
    print("FFmpeg audio conversion stderr:", result.stderr)

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

def generate_srt_subtitles(transcription_data, srt_path):
    # Grouper les mots en phrases
    phrases = []
    current_phrase = []
    current_start = None
    
    for word_info in transcription_data:
        if not current_start:
            current_start = word_info['start']
        
        current_phrase.append(word_info['word'])
        
        # Séparer les phrases tous les 5-6 mots ou si pause détectée
        if len(current_phrase) >= 5 or (word_info.get('pause', False)):
            phrases.append({
                'start': current_start,
                'end': word_info['end'],
                'text': ' '.join(current_phrase)
            })
            current_phrase = []
            current_start = None
    
    # Ajouter la dernière phrase si nécessaire
    if current_phrase:
        phrases.append({
            'start': current_start,
            'end': transcription_data[-1]['end'],
            'text': ' '.join(current_phrase)
        })
    
    # Écrire le fichier SRT
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, phrase in enumerate(phrases, start=1):
            # Format du timestamp pour SRT
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

def generate_video_with_subtitles_opencv(image_paths, audio_path, srt_path, output_path, fps=24):
    """
    1) Lit les sous-titres SRT et l'audio (pour sa durée).
    2) Répartit les images sur la durée totale de l'audio.
    3) Crée une liste de frames (en copiant l'image à chaque fois).
    4) Incruste les sous-titres dans les frames.
    5) Écrit une vidéo muette (silent_video.mp4).
    6) Fusionne ensuite l'audio et la vidéo pour produire output_path.
    """
    # --- 1) Lire les sous-titres et l'audio
    subtitles = read_srt(srt_path)
    audio = AudioSegment.from_file(audio_path)  # WAV, MP3, etc.
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000.0

    # --- 2) Calculer le nombre de frames total et la répartition
    total_frames = int(total_duration_sec * fps)
    if not image_paths:
        raise ValueError("Aucune image fournie dans image_paths.")
    segment_duration_sec = total_duration_sec / len(image_paths)

    # --- 3) Déterminer la taille de la vidéo à partir de la première image
    first_img = cv2.imread(image_paths[0])
    height, width, _ = first_img.shape

    # Préparer l'écriture d'une vidéo muette (temporaire)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    silent_video_path = output_path.replace('.mp4', '_silent.mp4')
    out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))

    # Générer la liste de frames (copie par frame pour éviter que le texte reste incrusté)
    frames_list = []
    for img_path in image_paths:
        img_original = cv2.imread(img_path)
        # Si nécessaire, on peut redimensionner ici :
        # img_original = cv2.resize(img_original, (width, height), interpolation=cv2.INTER_AREA)

        # Nombre de frames à générer pour cette image
        frames_for_this_image = int(segment_duration_sec * fps)

        for _ in range(frames_for_this_image):
            frames_list.append(img_original.copy())

    # Ajuster si on n'a pas exactement total_frames
    if len(frames_list) < total_frames:
        # On complète avec la dernière image
        while len(frames_list) < total_frames:
            frames_list.append(img_original.copy())
    elif len(frames_list) > total_frames:
        # On tronque
        frames_list = frames_list[:total_frames]

    # --- 4) Incruster les sous-titres sur chaque frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # blanc
    offset_y = 50  # marge depuis le bas

    for frame_index, frame in enumerate(frames_list):
        current_time = frame_index / fps  # temps en secondes
        # Chercher le sous-titre à afficher
        current_subtitle = None
        for start, end, text in subtitles:
            if start <= current_time < end:
                current_subtitle = text
                break

        if current_subtitle:
            # Calculer la position X pour centrer
            text_size = cv2.getTextSize(current_subtitle, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - offset_y

            # Dessiner un contour noir plus épais
            cv2.putText(frame, current_subtitle, (text_x, text_y),
                        font, font_scale, (0, 0, 0), font_thickness + 2)
            # Dessiner le texte en blanc
            cv2.putText(frame, current_subtitle, (text_x, text_y),
                        font, font_scale, font_color, font_thickness)

        # Écrire la frame dans la vidéo muette
        out.write(frame)

    out.release()  # fermeture de la vidéo muette

    # --- 5) Fusionner la vidéo muette et l'audio
    merge_audio_and_video(silent_video_path, audio_path, output_path)

    # (Optionnel) On peut supprimer la vidéo muette temporaire
    if os.path.exists(silent_video_path):
        os.remove(silent_video_path)

    print(f"Vidéo finale générée : {output_path}")

def merge_audio_and_video(video_path, audio_path, output_path):
    """
    Fusionne la vidéo muette (video_path) et l'audio (audio_path) dans output_path
    en utilisant ffmpeg. Conserve la piste vidéo telle quelle et encode l'audio en AAC.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',  # coupe si l'audio est plus long que la vidéo
        output_path
    ]
    subprocess.run(cmd, check=True)

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

    transcription_data = transcribe_audio_in_chunks(wav_path, chunk_duration_ms=30000)

    srt_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.srt")
    generate_srt_subtitles(transcription_data, srt_path)

    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
    generate_video_with_subtitles_opencv(image_paths, wav_path, srt_path, output_path)

    if not os.path.exists(output_path):
        print(f"Erreur : La vidéo {output_path} n'a pas été créée.")
        return jsonify({'error': 'Video file not created'}), 500

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)