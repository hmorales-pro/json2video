/**
 * json2video — serveur HTTP
 *
 * Routes :
 *   GET  /                       healthcheck
 *   GET  /templates              liste des templates intégrés
 *   GET  /formats                liste des presets de formats
 *   POST /render                 nouvelle API de composition (1→N médias)
 *   POST /generate-video         API legacy (1 image + 1 audio)
 *   GET  /download/:filename     téléchargement protégé
 */
const express = require("express");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const multer = require("multer");
const fs = require("fs/promises");
const path = require("path");
const crypto = require("crypto");
const { v4: uuidv4 } = require("uuid");
require("dotenv").config();

const { resolveConfig, listTemplates } = require("./lib/templates");
const { listFormats } = require("./lib/formats");
const { validateConfig } = require("./lib/validate");
const { render } = require("./lib/render");
const { transcribeSrt, transcribeWordLevel } = require("./lib/transcribe");
const { defaultQueue } = require("./lib/queue");
const { runPreflight, logPreflight } = require("./lib/preflight");

// ---------------------------------------------------------------
// Config
// ---------------------------------------------------------------
const PORT = parseInt(process.env.NODE_PORT || "3000", 10);
const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || `http://localhost:${PORT}`;
const SECRET_API_KEY = process.env.SECRET_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MAX_UPLOAD_MB = parseInt(process.env.MAX_UPLOAD_MB || "200", 10);

if (!SECRET_API_KEY) {
  console.error("FATAL: SECRET_API_KEY manquant dans .env");
  process.exit(1);
}
if (!OPENAI_API_KEY) {
  console.error("FATAL: OPENAI_API_KEY manquant dans .env");
  process.exit(1);
}

const transcribeFns = {
  srt: transcribeSrt(OPENAI_API_KEY),
  wordLevel: transcribeWordLevel(OPENAI_API_KEY),
};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------
const ensureDirectoryExists = async (dir) => {
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (e) {
    console.error(`mkdir ${dir}:`, e.message);
  }
};

function cleanupFiles(paths) {
  Promise.allSettled(
    paths.map((p) =>
      fs.unlink(p).catch((e) => {
        if (e.code !== "ENOENT") console.warn(`unlink ${p}:`, e.message);
      })
    )
  );
}

// ---------------------------------------------------------------
// App
// ---------------------------------------------------------------
const app = express();
app.disable("x-powered-by");

// Derrière un reverse proxy (Traefik/Caddy/Nginx), on doit faire confiance
// aux headers X-Forwarded-For pour identifier la vraie IP client.
// La valeur 1 = on fait confiance UNIQUEMENT au 1er proxy de la chaîne
// (Traefik chez Dokploy). C'est requis pour que express-rate-limit
// fonctionne correctement.
app.set("trust proxy", 1);

app.use(helmet());
app.use(express.json({ limit: "1mb" }));

app.use(
  rateLimit({
    windowMs: 60 * 1000,
    max: 20,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Trop de requêtes, réessayez dans 1 minute." },
  })
);

function requireApiKey(req, res, next) {
  const provided = req.headers["x-api-key"] || req.query.api_key;
  if (!provided) return res.status(401).json({ error: "API key manquante" });
  const a = Buffer.from(String(provided));
  const b = Buffer.from(SECRET_API_KEY);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    return res.status(401).json({ error: "API key invalide" });
  }
  return next();
}

// ---------------------------------------------------------------
// Multer
// ---------------------------------------------------------------
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024, files: 60 },
  fileFilter: (req, file, cb) => {
    const ok =
      file.mimetype.startsWith("image/") ||
      file.mimetype.startsWith("audio/") ||
      file.mimetype.startsWith("video/");
    if (!ok) return cb(new Error("Type de fichier non supporté (image/audio/video uniquement)"));
    cb(null, true);
  },
});

// ---------------------------------------------------------------
// Routes publiques
// ---------------------------------------------------------------
app.get("/", (req, res) => res.json({ status: "ok", service: "json2video" }));

app.get("/templates", requireApiKey, (req, res) => {
  res.json({ templates: listTemplates() });
});

app.get("/formats", requireApiKey, (req, res) => {
  res.json({ formats: listFormats() });
});

// ---------------------------------------------------------------
// POST /render — la nouvelle API
// ---------------------------------------------------------------
//
// multipart/form-data :
//   - "config" : (string JSON, optionnel) la config de rendu
//   - "audio"  : (file, optionnel) audio principal
//   - "media_0", "media_1", ... "media_N" : fichiers médias dans l'ordre
//
// Si "config" est absent, on utilise par défaut le template "reels-karaoke"
// et on aligne config.media sur le nombre de fichiers media_* reçus.
//
/**
 * Parse les fichiers multipart en {audio, music, media[]}.
 * Tolère deux conventions :
 *   - nommée : fieldname = "audio", "music", "media_0", "media_1"...
 *   - libre : détection MIME (premier audio = voix, premier video/image = média)
 */
function _parseUploads(allFiles) {
  let audioFile = allFiles.find((f) => f.fieldname === "audio");
  let musicFile = allFiles.find((f) => f.fieldname === "music");
  const mediaFiles = allFiles
    .filter((f) => f !== audioFile && f !== musicFile && /^media(_\d+)?$/.test(f.fieldname))
    .sort((a, b) => {
      const idx = (f) => {
        const m = f.fieldname.match(/^media_(\d+)$/);
        return m ? parseInt(m[1], 10) : 0;
      };
      return idx(a) - idx(b);
    });

  // Fallback : si aucun fieldname nommé, on essaie de déduire depuis MIME
  if (!audioFile && !mediaFiles.length) {
    const audios = allFiles.filter((f) => f.mimetype.startsWith("audio/"));
    audioFile = audios[0];
    musicFile = audios[1]; // si 2 audios, le 2e est considéré comme musique
    for (const f of allFiles) {
      if (f === audioFile || f === musicFile) continue;
      if (f.mimetype.startsWith("image/") || f.mimetype.startsWith("video/")) {
        mediaFiles.push(f);
      }
    }
  }
  return { audioFile, musicFile, mediaFiles };
}

async function _buildValidatedConfig(req, mediaFiles) {
  let userConfig = {};
  if (req.body?.config) {
    try {
      userConfig = JSON.parse(req.body.config);
    } catch {
      throw Object.assign(new Error("Champ 'config' invalide (JSON malformé)."), { status: 400 });
    }
  }
  if (!userConfig.template && !userConfig.format) {
    userConfig.template = "reels-karaoke";
  }
  if (!Array.isArray(userConfig.media)) {
    userConfig.media = mediaFiles.map(() => ({}));
  } else if (userConfig.media.length !== mediaFiles.length) {
    throw Object.assign(
      new Error(`Mismatch: ${mediaFiles.length} fichier(s) reçu(s) vs ${userConfig.media.length} entrée(s) config.media.`),
      { status: 400 }
    );
  }
  return validateConfig(resolveConfig(userConfig));
}

app.post(
  "/render",
  requireApiKey,
  upload.any(),
  async (req, res) => {
    await ensureDirectoryExists("uploads");
    await ensureDirectoryExists("outputs");
    const allFiles = req.files || [];
    const tempInputs = allFiles.map((f) => f.path);

    try {
      const { audioFile, musicFile, mediaFiles } = _parseUploads(allFiles);
      if (!mediaFiles.length) {
        return res.status(400).json({
          error: "Au moins un fichier média (media_0, media_1...) est requis.",
        });
      }
      const validated = await _buildValidatedConfig(req, mediaFiles);
      const outputPath = path.resolve("outputs", `${uuidv4()}_final.mp4`);
      const { totalDurationSec } = await render({
        config: validated,
        mediaFiles,
        audioFile,
        musicFile,
        transcribeFns,
        outputPath,
        tmpDir: "uploads",
      });
      return res.json({
        downloadUrl: `${PUBLIC_BASE_URL}/download/${path.basename(outputPath)}`,
        duration_sec: totalDurationSec,
        format: `${validated.format.width}x${validated.format.height}`,
        fps: validated.fps,
        subtitle_mode: validated.subtitle.mode,
        media_count: mediaFiles.length,
        transitions: validated.transitions.type,
        overlays_count: validated.overlays.length,
      });
    } catch (err) {
      console.error("/render KO:", err.message);
      const status = err.status || 500;
      return res.status(status).json({
        error: status === 500 ? "Erreur interne" : err.message,
        ...(status !== 500 && { details: err.message }),
      });
    } finally {
      cleanupFiles(tempInputs);
    }
  }
);

// ---------------------------------------------------------------
// POST /render/async — version queue (job_id)
// ---------------------------------------------------------------
app.post(
  "/render/async",
  requireApiKey,
  upload.any(),
  async (req, res) => {
    await ensureDirectoryExists("uploads");
    await ensureDirectoryExists("outputs");
    const allFiles = req.files || [];
    try {
      const { audioFile, musicFile, mediaFiles } = _parseUploads(allFiles);
      if (!mediaFiles.length) {
        cleanupFiles(allFiles.map((f) => f.path));
        return res.status(400).json({ error: "Au moins un fichier média requis." });
      }
      const validated = await _buildValidatedConfig(req, mediaFiles);
      const outputPath = path.resolve("outputs", `${uuidv4()}_final.mp4`);
      const tempInputs = allFiles.map((f) => f.path);

      const jobId = defaultQueue.enqueue(
        async () => {
          try {
            const { totalDurationSec } = await render({
              config: validated, mediaFiles, audioFile, musicFile,
              transcribeFns, outputPath, tmpDir: "uploads",
            });
            return {
              downloadUrl: `${PUBLIC_BASE_URL}/download/${path.basename(outputPath)}`,
              duration_sec: totalDurationSec,
              media_count: mediaFiles.length,
            };
          } finally {
            cleanupFiles(tempInputs);
          }
        },
        {
          media_count: mediaFiles.length,
          format: `${validated.format.width}x${validated.format.height}`,
          subtitle_mode: validated.subtitle.mode,
        }
      );
      return res.status(202).json({ job_id: jobId, status_url: `${PUBLIC_BASE_URL}/jobs/${jobId}` });
    } catch (err) {
      cleanupFiles(allFiles.map((f) => f.path));
      console.error("/render/async KO:", err.message);
      const status = err.status || 500;
      return res.status(status).json({
        error: status === 500 ? "Erreur interne" : err.message,
      });
    }
  }
);

app.get("/jobs/:id", requireApiKey, (req, res) => {
  const job = defaultQueue.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job inconnu." });
  res.json(job);
});

app.get("/jobs", requireApiKey, (req, res) => {
  res.json({ jobs: defaultQueue.list({ status: req.query.status }) });
});

// ---------------------------------------------------------------
// POST /generate-video — legacy (1 image + 1 audio)
// ---------------------------------------------------------------
app.post(
  "/generate-video",
  requireApiKey,
  upload.array("files"),
  async (req, res) => {
    await ensureDirectoryExists("uploads");
    await ensureDirectoryExists("outputs");
    const files = req.files || [];
    const tempInputs = files.map((f) => f.path);

    try {
      const images = files.filter((f) => f.mimetype.startsWith("image"));
      const audio = files.find((f) => f.mimetype.startsWith("audio"));
      if (!images.length || !audio) {
        return res
          .status(400)
          .json({ error: "Au moins une image + un fichier audio requis." });
      }

      const karaoke = String(req.body?.karaoke ?? req.query?.karaoke ?? "false")
        .toLowerCase() === "true";

      // On passe par /render en interne : 1 image, 9:16 karaoke par défaut.
      const userConfig = {
        template: karaoke ? "reels-karaoke" : "youtube-captions",
        media: [{ duration: 5.0, fit: "cover" }],
      };
      const validated = validateConfig(resolveConfig(userConfig));

      const outputPath = path.resolve("outputs", `${uuidv4()}_final.mp4`);
      await render({
        config: validated,
        mediaFiles: [images[0]],
        audioFile: audio,
        transcribeFns,
        outputPath,
        tmpDir: "uploads",
      });

      return res.json({
        downloadUrl: `${PUBLIC_BASE_URL}/download/${path.basename(outputPath)}`,
      });
    } catch (err) {
      console.error("/generate-video KO:", err.message);
      return res
        .status(err.status || 500)
        .json({ error: "Erreur génération vidéo", details: err.message });
    } finally {
      cleanupFiles(tempInputs);
    }
  }
);

// ---------------------------------------------------------------
// Download
// ---------------------------------------------------------------
const SAFE_NAME = /^[A-Za-z0-9._-]+$/;
app.get("/download/:filename", requireApiKey, (req, res) => {
  const filename = req.params.filename;
  if (!SAFE_NAME.test(filename)) {
    return res.status(400).json({ error: "Nom de fichier invalide" });
  }
  return res.sendFile(path.resolve("outputs", filename));
});

// ---------------------------------------------------------------
// Error handler
// ---------------------------------------------------------------
app.use((err, req, res, _next) => {
  console.error("Unhandled:", err);
  res.status(err.status || 500).json({
    error: err.status ? err.message : "Erreur interne",
  });
});

app.listen(PORT, async () => {
  console.log(`Serveur démarré sur http://localhost:${PORT}`);
  try {
    const probes = await runPreflight();
    logPreflight(probes);
  } catch (e) {
    console.warn("Preflight KO (non bloquant) :", e.message);
  }
});
