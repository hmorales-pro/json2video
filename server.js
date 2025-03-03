const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const fs = require("fs/promises");
const fsSync = require("fs");
const path = require("path");
const { v4: uuidv4 } = require("uuid");
const axios = require("axios");
const FormData = require("form-data");
const { exec } = require("child_process");
require("dotenv").config();

const app = express();
const port = 3000;

// Valeur de l'offset pour ajuster la synchronisation initiale des sous-titres
const subtitleOffset = 0.0; // Ajustez si nécessaire

// Configuration de multer pour stocker temporairement les fichiers
const upload = multer({ dest: "uploads/" });

// Création (si nécessaire) des dossiers "uploads" et "outputs"
const ensureDirectoryExists = async (dir) => {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (error) {
    console.error(`Erreur lors de la création du dossier ${dir}:`, error.message);
  }
};

// Fonction pour convertir SRT en ASS en appelant le script Python
const convertSrtToAss = async (srtPath, assPath) => {
    return new Promise((resolve, reject) => {
        exec(`/Users/hugomorales/miniconda3/bin/python3 convert_srt_to_ass.py "${srtPath}" "${assPath}"`, 
        (error, stdout, stderr) => {
            if (error) {
                console.error("Erreur lors de la conversion SRT vers ASS :", stderr);
                return reject(error);
            }
            console.log("Conversion SRT vers ASS réussie :", stdout);
            resolve();
        });
    });
};

// Fonction pour transcrire l'audio en sous-titres SRT via l'API OpenAI
const transcribeAudio = async (audioPath, originalName) => {
    try {
        if (!process.env.OPENAI_API_KEY) {
            throw new Error("Clé API OpenAI manquante. Vérifiez votre fichier .env.");
        }

        const formData = new FormData();
        formData.append("file", fsSync.createReadStream(audioPath), originalName || "audio.mp3");
        formData.append("model", "whisper-1");
        formData.append("language", "fr");
        formData.append("response_format", "srt");

        const contentLength = await new Promise((resolve, reject) => {
            formData.getLength((err, length) => {
                if (err) reject(err);
                else resolve(length);
            });
        });

        const response = await axios.post("https://api.openai.com/v1/audio/transcriptions", formData, {
            headers: {
                "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
                ...formData.getHeaders(),
                "Content-Length": contentLength,
            },
        });

        return response.data; // Contenu SRT généré par Whisper
    } catch (error) {
        console.error("Erreur lors de la transcription OpenAI:", error.response?.data || error.message);
        throw new Error("Impossible de transcrire l'audio : " + (error.response?.data?.error?.message || error.message));
    }
};


// Endpoint pour générer la vidéo avec sous-titres animés
app.post("/generate-video", upload.array("files"), async (req, res) => {
    try {
        await ensureDirectoryExists("uploads");
        await ensureDirectoryExists("outputs");

        const files = req.files;
        if (!files || files.length < 2) {
            return res.status(400).json({ error: "Envoyez au moins une image et un audio." });
        }

        const images = files.filter(file => file.mimetype.startsWith("image"));
        const audio = files.find(file => file.mimetype.startsWith("audio"));
        if (!audio) {
            return res.status(400).json({ error: "Fichier audio manquant." });
        }

        const audioPath = path.resolve(audio.path);
        const srtPath = `uploads/${uuidv4()}.srt`;
        const assPath = `uploads/${uuidv4()}.ass`;

        // Transcription de l'audio en sous-titres SRT avec OpenAI
        const srtContent = await transcribeAudio(audioPath, audio.originalname);
        await fs.writeFile(srtPath, srtContent, "utf8");

        // Conversion en ASS avec animation
        await convertSrtToAss(srtPath, assPath);

        const finalVideo = `outputs/${uuidv4()}_final.mp4`;
        await new Promise((resolve, reject) => {
            ffmpeg()
                .input(images[0].path)
                .input(audioPath)
                .outputOptions(["-vf subtitles=" + assPath])
                .save(finalVideo)
                .on("start", commandLine => console.log("Commande FFmpeg :", commandLine))
                .on("end", resolve)
                .on("error", reject);
        });

        await fs.unlink(srtPath);
        await fs.unlink(assPath);
        images.forEach(img => fs.unlink(img.path));
        await fs.unlink(audioPath);

        res.json({ downloadUrl: `http://localhost:${port}/download/${path.basename(finalVideo)}` });
    } catch (error) {
        console.error("Erreur lors de la génération de la vidéo :", error.message);
        res.status(500).json({ error: "Erreur lors de la génération de la vidéo.", details: error.message });
    }
});

// Démarrer le serveur
app.listen(port, () => {
    console.log(`Serveur démarré sur http://localhost:${port}`);
});
