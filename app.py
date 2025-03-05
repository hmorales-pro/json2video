from flask import Flask, request, jsonify, send_file, send_from_directory
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

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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

def draw_text_pil(frame, text, x, y, font_path="DejaVuSans.ttf", font_size=24,
                  text_color=(255, 255, 255), stroke_color=(0, 0, 0), stroke_width=2):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Erreur: police '{font_path}' non trouvée. Utilisation de la police par défaut.")
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_bgr

def merge_audio_and_video(video_path, audio_path, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'libx264',       # Re-encoder en H264
        '-preset', 'veryfast',
        '-crf', '23',
        '-c:a', 'aac',           # Audio en AAC
        '-shortest',
        output_path
    ]
    subprocess.run(cmd, check=True)

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
            # On fige la ligne courante, et on démarre une nouvelle
            lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def generate_video_with_subtitles_opencv(image_paths, audio_path, srt_path, output_path,
                                         fps=24, font_path="DejaVuSans.ttf", font_size=24):
    subtitles = read_srt(srt_path)
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000.0

    if not image_paths:
        raise ValueError("Aucune image fournie.")
    first_img = cv2.imread(image_paths[0])
    height, width, _ = first_img.shape

    total_frames = int(total_duration_sec * fps)
    segment_duration_sec = total_duration_sec / len(image_paths)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    silent_video_path = output_path.replace('.mp4', '_silent.mp4')
    out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))

    # Prépare la liste de frames
    frames_list = []
    for img_path in image_paths:
        img_original = cv2.imread(img_path)
        frames_for_this_image = int(segment_duration_sec * fps)
        for _ in range(frames_for_this_image):
            frames_list.append(img_original.copy())

    # Ajuste si trop ou pas assez de frames
    if len(frames_list) < total_frames:
        while len(frames_list) < total_frames:
            frames_list.append(img_original.copy())
    elif len(frames_list) > total_frames:
        frames_list = frames_list[:total_frames]

    # Pour chaque frame, on cherche le sous-titre correspondant, on le découpe en lignes, on dessine
    for frame_index, frame in enumerate(frames_list):
        current_time = frame_index / fps
        current_subtitle = None
        for start, end, text in subtitles:
            if start <= current_time < end:
                current_subtitle = text
                break

        if current_subtitle:
            dummy_img = Image.new("RGB", (width, height))
            draw_dummy = ImageDraw.Draw(dummy_img)
            try:
                font_pil = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Erreur: police '{font_path}' non trouvée pour le dimensionnement. Utilisation de la police par défaut.")
                font_pil = ImageFont.load_default()

            max_text_width = int(width * 0.8)
            lines = wrap_text_by_width(draw_dummy, current_subtitle, font_pil, max_text_width)

            # Calculer la hauteur totale de toutes les lignes
            total_text_height = 0
            line_heights = []
            for line in lines:
                bbox_line = draw_dummy.textbbox((0, 0), line, font=font_pil)
                line_height = bbox_line[3] - bbox_line[1]
                line_heights.append(line_height)
                total_text_height += line_height

            # Centrage vertical (au milieu). 
            # Si tu préfères en bas, fais: current_y = height - offset_y - total_text_height
            current_y = (height - total_text_height) // 2

            # Dessin de chaque ligne
            for i, one_line in enumerate(lines):
                bbox_line = draw_dummy.textbbox((0, 0), one_line, font=font_pil)
                line_w = bbox_line[2] - bbox_line[0]
                line_h = line_heights[i]

                # Centrage horizontal
                text_x = (width - line_w) // 2

                # Dessin sur la frame
                frame = draw_text_pil(
                    frame,
                    one_line,
                    text_x,
                    current_y,
                    font_path=font_path,
                    font_size=font_size,
                    text_color=(255, 255, 255),
                    stroke_color=(0, 0, 0),
                    stroke_width=2
                )

                # Décaler vers le bas pour la prochaine ligne
                current_y += line_h

        # On écrit la frame, qu'il y ait un sous-titre ou non
        out.write(frame)

    out.release()

    # Fusion audio + vidéo
    merge_audio_and_video(silent_video_path, audio_path, output_path)
    if os.path.exists(silent_video_path):
        os.remove(silent_video_path)

    print(f"Vidéo finale générée : {output_path}")

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    # On force le téléchargement ou la lecture en tant que vidéo mp4
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

    transcription_data = transcribe_audio_in_chunks(wav_path, chunk_duration_ms=30000)

    srt_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.srt")
    generate_srt_subtitles(transcription_data, srt_path)

    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}.mp4")
    generate_video_with_subtitles_opencv(image_paths, wav_path, srt_path, output_path,
                                         fps=24,
                                         font_path="DejaVuSans.ttf",
                                         font_size=48)  # <--- Augmente la taille de la police si tu veux

    if not os.path.exists(output_path):
        print(f"Erreur : La vidéo {output_path} n'a pas été créée.")
        return jsonify({'error': 'Video file not created'}), 500

    # Construire l'URL de téléchargement
    filename = os.path.basename(output_path)
    file_url = request.host_url + 'outputs/' + filename

    return jsonify({'url': file_url})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
