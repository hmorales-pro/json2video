#!/usr/bin/env node
/**
 * Serveur MCP json2video.
 *
 * Expose le moteur de composition vidéo aux agents IA via le protocole
 * Model Context Protocol (transport stdio).
 *
 * Configuration côté agent (exemple Claude Desktop / Cowork) :
 *   {
 *     "mcpServers": {
 *       "json2video": {
 *         "command": "node",
 *         "args": ["/chemin/vers/json2video/mcp-server.js"],
 *         "env": {
 *           "OPENAI_API_KEY": "sk-...",
 *           "JSON2VIDEO_OUTPUT_DIR": "/Users/hugo/Videos/json2video"
 *         }
 *       }
 *     }
 *   }
 *
 * Tools exposés :
 *   - json2video_list_templates  : liste des templates disponibles
 *   - json2video_list_formats    : liste des presets de formats
 *   - json2video_probe_media     : info technique sur un fichier média
 *   - json2video_render          : génère une vidéo (1→N médias + audio + config)
 */
const { Server } = require("@modelcontextprotocol/sdk/server/index.js");
const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
const {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} = require("@modelcontextprotocol/sdk/types.js");

const fs = require("fs/promises");
const fsSync = require("fs");
const path = require("path");
const os = require("os");
const { v4: uuidv4 } = require("uuid");
const https = require("https");
const http = require("http");
const { URL } = require("url");
require("dotenv").config();

const { resolveConfig, listTemplates } = require("./lib/templates");
const { listFormats } = require("./lib/formats");
const { validateConfig } = require("./lib/validate");
const { render, probeMedia } = require("./lib/render");
const { transcribeSrt, transcribeWordLevel } = require("./lib/transcribe");

// ---------------------------------------------------------------
// Config
// ---------------------------------------------------------------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DEFAULT_OUTPUT_DIR =
  process.env.JSON2VIDEO_OUTPUT_DIR ||
  path.join(os.homedir(), "json2video-outputs");

const transcribeFns = OPENAI_API_KEY
  ? { srt: transcribeSrt(OPENAI_API_KEY), wordLevel: transcribeWordLevel(OPENAI_API_KEY) }
  : null;

// ---------------------------------------------------------------
// Helpers : récupérer un média à partir d'une URL ou d'un chemin local
// ---------------------------------------------------------------
async function _resolveMediaSource(src, tmpDir) {
  if (!src || typeof src !== "string") {
    throw new Error("media source: chaîne attendue (url ou chemin local).");
  }
  // file://...
  if (src.startsWith("file://")) {
    const p = new URL(src).pathname;
    return _wrapAsMulterFile(p);
  }
  // http(s)://
  if (/^https?:\/\//.test(src)) {
    const downloaded = await _download(src, tmpDir);
    return _wrapAsMulterFile(downloaded);
  }
  // Chemin local absolu ou relatif
  return _wrapAsMulterFile(path.resolve(src));
}

function _wrapAsMulterFile(filePath) {
  if (!fsSync.existsSync(filePath)) {
    throw new Error(`Fichier introuvable : ${filePath}`);
  }
  const ext = path.extname(filePath).toLowerCase();
  const mimeMap = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg",
  };
  return {
    path: filePath,
    originalname: path.basename(filePath),
    mimetype: mimeMap[ext] || "application/octet-stream",
  };
}

function _download(url, tmpDir) {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith("https") ? https : http;
    const ext = path.extname(new URL(url).pathname) || ".bin";
    const target = path.join(tmpDir, `${uuidv4()}${ext}`);
    const file = fsSync.createWriteStream(target);
    lib
      .get(url, (resp) => {
        if (resp.statusCode && resp.statusCode >= 300 && resp.statusCode < 400 && resp.headers.location) {
          file.close();
          fsSync.unlinkSync(target);
          return _download(resp.headers.location, tmpDir).then(resolve, reject);
        }
        if (resp.statusCode !== 200) {
          file.close();
          fsSync.unlinkSync(target);
          return reject(new Error(`HTTP ${resp.statusCode} sur ${url}`));
        }
        resp.pipe(file);
        file.on("finish", () => file.close(() => resolve(target)));
      })
      .on("error", (err) => {
        try { fsSync.unlinkSync(target); } catch {}
        reject(err);
      });
  });
}

// ---------------------------------------------------------------
// Définition du serveur MCP + tools
// ---------------------------------------------------------------
const server = new Server(
  {
    name: "json2video",
    version: "1.0.0",
  },
  { capabilities: { tools: {} } }
);

const TOOLS = [
  {
    name: "json2video_list_templates",
    description:
      "Liste les templates de composition vidéo disponibles (reels-karaoke, youtube-captions, square-quote, tiktok-storytime). Chaque template définit format, fps, mode de sous-titres et style par défaut.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "json2video_list_formats",
    description:
      "Liste les presets de formats vidéo (9:16, 16:9, 1:1, 4:5, 21:9) avec leurs dimensions.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "json2video_probe_media",
    description:
      "Retourne les métadonnées techniques d'un fichier média (largeur, hauteur, durée, présence audio/vidéo). Accepte un chemin local ou une URL http(s).",
    inputSchema: {
      type: "object",
      properties: {
        source: {
          type: "string",
          description: "Chemin local ou URL du fichier à analyser.",
        },
      },
      required: ["source"],
      additionalProperties: false,
    },
  },
  {
    name: "json2video_render",
    description:
      "Génère une vidéo MP4 à partir d'un ou plusieurs médias (images ou vidéos), d'un audio (voix-off) optionnel, d'une musique de fond optionnelle, et d'une config de rendu. " +
      "Supporte : 5 formats standard + custom ; sous-titres mode none/captions/karaoké ; transitions xfade ; mix audio voice+music (avec ducking) ; overlays texte. " +
      "Templates intégrés : reels-karaoke (Reels/TikTok 9:16 mot-à-mot), youtube-captions (16:9 captions), square-quote (1:1 citation), tiktok-storytime (9:16 karaoké centré). " +
      "Renvoie le chemin local du MP4 généré.",
    inputSchema: {
      type: "object",
      properties: {
        media: {
          type: "array",
          minItems: 1,
          maxItems: 50,
          description:
            "Liste des médias (chemin local ou URL http(s)). Image OU vidéo. Au moins 1, max 50.",
          items: {
            type: "object",
            properties: {
              source: { type: "string", description: "URL ou chemin du fichier." },
              duration: {
                type: "number",
                description: "Pour une image : durée d'affichage en secondes (défaut 3.0).",
              },
              fit: {
                type: "string",
                enum: ["cover", "contain", "stretch"],
                description: "Mode de cadrage. cover=crop, contain=letterbox, stretch=déforme.",
              },
              trim_start: { type: "number", description: "Vidéo: début du trim (s)." },
              trim_end: { type: "number", description: "Vidéo: fin du trim (s)." },
            },
            required: ["source"],
            additionalProperties: false,
          },
        },
        audio: {
          type: "string",
          description: "Chemin/URL de l'audio principal (voix-off). Requis pour générer des sous-titres.",
        },
        music: {
          type: "string",
          description: "Chemin/URL d'une musique de fond optionnelle.",
        },
        template: {
          type: "string",
          enum: ["reels-karaoke", "youtube-captions", "square-quote", "tiktok-storytime"],
          description: "Template pré-configuré. Si absent ET pas de format, défaut = reels-karaoke.",
        },
        format: {
          oneOf: [
            { type: "string", enum: ["9:16", "16:9", "1:1", "4:5", "21:9"] },
            {
              type: "object",
              properties: {
                width: { type: "integer", minimum: 16, maximum: 4096 },
                height: { type: "integer", minimum: 16, maximum: 4096 },
              },
              required: ["width", "height"],
            },
          ],
          description: "Format vidéo (preset ou dimensions custom paires).",
        },
        fps: { type: "integer", minimum: 12, maximum: 60, description: "Frame rate (12-60)." },
        subtitle: {
          type: "object",
          properties: {
            mode: {
              type: "string",
              enum: ["none", "captions", "karaoke"],
              description: "none = pas de sous-titres ; captions = blocs ; karaoke = mot-à-mot animé.",
            },
            language: {
              type: "string",
              description: "Code langue BCP-47 court (fr, en, fr-FR...).",
            },
          },
        },
        transitions: {
          type: "object",
          properties: {
            type: {
              type: "string",
              enum: ["none", "fade", "fadeblack", "fadewhite", "dissolve",
                     "wipeleft", "wiperight", "wipeup", "wipedown",
                     "slideleft", "slideright", "slideup", "slidedown",
                     "circlecrop", "radial", "zoomin"],
              description: "Type de transition entre médias consécutifs.",
            },
            duration: { type: "number", minimum: 0.1, maximum: 2.0 },
          },
        },
        audio_mix: {
          type: "object",
          description: "Mixage musique de fond + voix-off.",
          properties: {
            music_volume: { type: "number", minimum: 0, maximum: 2 },
            voice_volume: { type: "number", minimum: 0, maximum: 2 },
            ducking: {
              type: "boolean",
              description: "true = la musique baisse automatiquement quand la voix est présente.",
            },
          },
        },
        overlays: {
          type: "array",
          maxItems: 20,
          description: "Textes incrustés à des moments précis (titres, légendes...).",
          items: {
            type: "object",
            properties: {
              text: { type: "string" },
              start: { type: "number" },
              end: { type: "number" },
              x: {
                oneOf: [
                  { type: "string", enum: ["left", "center", "right"] },
                  { type: "integer" },
                ],
              },
              y: {
                oneOf: [
                  { type: "string", enum: ["top", "middle", "bottom"] },
                  { type: "integer" },
                ],
              },
              fontsize: { type: "integer", minimum: 8, maximum: 300 },
              color: { type: "string", description: "Hex #RRGGBB" },
              bg_color: { type: "string", description: "Hex #RRGGBB" },
              bg_opacity: { type: "number", minimum: 0, maximum: 1 },
            },
            required: ["text"],
            additionalProperties: false,
          },
        },
        output_dir: {
          type: "string",
          description: `Répertoire où écrire le MP4 (défaut: $JSON2VIDEO_OUTPUT_DIR ou ~/json2video-outputs).`,
        },
      },
      required: ["media"],
      additionalProperties: false,
    },
  },
];

// Handlers
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;
  try {
    switch (name) {
      case "json2video_list_templates":
        return _ok({ templates: listTemplates() });

      case "json2video_list_formats":
        return _ok({ formats: listFormats() });

      case "json2video_probe_media": {
        const file = await _resolveMediaSource(args.source, os.tmpdir());
        const info = await probeMedia(file.path);
        return _ok(info);
      }

      case "json2video_render": {
        if (!transcribeFns && (args.subtitle?.mode ?? "none") !== "none") {
          throw new Error(
            "OPENAI_API_KEY manquant : impossible de générer des sous-titres. " +
            "Définissez OPENAI_API_KEY dans l'environnement du serveur MCP, " +
            "ou utilisez subtitle.mode='none'."
          );
        }
        return await _renderTool(args);
      }

      default:
        throw new Error(`Tool inconnu : ${name}`);
    }
  } catch (err) {
    return _err(err.message || String(err));
  }
});

function _ok(payload) {
  return {
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
  };
}
function _err(message) {
  return {
    isError: true,
    content: [{ type: "text", text: `Erreur: ${message}` }],
  };
}

// ---------------------------------------------------------------
// Rendu (avec récupération des fichiers depuis URL/chemin)
// ---------------------------------------------------------------
async function _renderTool(args) {
  const outDir = path.resolve(args.output_dir || DEFAULT_OUTPUT_DIR);
  await fs.mkdir(outDir, { recursive: true });
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "json2video-mcp-"));

  const downloadedPaths = [];
  try {
    // Récupérer audio + music + médias
    const audioFile = args.audio
      ? await _resolveMediaSource(args.audio, tmpDir)
      : null;
    const musicFile = args.music
      ? await _resolveMediaSource(args.music, tmpDir)
      : null;
    const mediaFiles = [];
    const mediaConfigs = [];
    for (const m of args.media) {
      const f = await _resolveMediaSource(m.source, tmpDir);
      mediaFiles.push(f);
      const { source, ...rest } = m;
      mediaConfigs.push(rest);
    }
    // Trace pour cleanup les téléchargements
    for (const f of [audioFile, musicFile, ...mediaFiles].filter(Boolean)) {
      if (f.path.startsWith(tmpDir)) downloadedPaths.push(f.path);
    }

    // Construire la config et valider
    const userConfig = {
      template: args.template,
      format: args.format,
      fps: args.fps,
      subtitle: args.subtitle,
      transitions: args.transitions,
      audio: args.audio_mix,
      overlays: args.overlays,
      media: mediaConfigs,
    };
    if (!userConfig.template && !userConfig.format) userConfig.template = "reels-karaoke";
    Object.keys(userConfig).forEach((k) => userConfig[k] === undefined && delete userConfig[k]);

    const validated = validateConfig(resolveConfig(userConfig));
    const outputPath = path.join(outDir, `${uuidv4()}.mp4`);

    const { totalDurationSec } = await render({
      config: validated,
      mediaFiles, audioFile, musicFile,
      transcribeFns,
      outputPath,
      tmpDir,
    });

    return _ok({
      output_path: outputPath,
      duration_sec: Number(totalDurationSec.toFixed(2)),
      format: `${validated.format.width}x${validated.format.height}`,
      fps: validated.fps,
      subtitle_mode: validated.subtitle.mode,
      transitions: validated.transitions.type,
      media_count: mediaFiles.length,
      has_voice: !!audioFile,
      has_music: !!musicFile,
      overlays_count: validated.overlays.length,
    });
  } finally {
    // Cleanup des téléchargements et du tmpDir
    for (const p of downloadedPaths) fs.unlink(p).catch(() => {});
    fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

// ---------------------------------------------------------------
// Démarrage
// ---------------------------------------------------------------
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // NB : on écrit sur stderr pour ne pas polluer le canal stdio MCP.
  console.error(`json2video MCP server prêt. Output dir: ${DEFAULT_OUTPUT_DIR}`);
}

main().catch((err) => {
  console.error("MCP server fatal:", err);
  process.exit(1);
});
