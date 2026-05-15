/**
 * Validation stricte de la config de rendu.
 *
 * Pourquoi c'est critique : la config arrive du client en JSON et finit
 * partiellement dans des arguments FFmpeg. Tout champ qui ne passe pas
 * dans une whitelist est rejeté pour éviter l'injection d'options.
 */
const { resolveFormat } = require("./formats");

const SUBTITLE_MODES = new Set(["none", "captions", "karaoke"]);
const FIT_MODES = new Set(["cover", "contain", "stretch"]);
const FFMPEG_PRESETS = new Set([
  "ultrafast", "superfast", "veryfast", "faster", "fast",
  "medium", "slow", "slower", "veryslow",
]);
// Whitelist xfade : on ne laisse passer que les transitions standards
// pour éviter qu'un type exotique fasse exploser le filter_complex.
const XFADE_TYPES = new Set([
  "none",
  "fade", "fadeblack", "fadewhite", "dissolve",
  "wipeleft", "wiperight", "wipeup", "wipedown",
  "slideleft", "slideright", "slideup", "slidedown",
  "smoothleft", "smoothright", "smoothup", "smoothdown",
  "circlecrop", "rectcrop", "distance", "radial", "zoomin",
]);
const AUDIO_HANDLING = new Set(["extend", "loop", "loop_all", "cut"]);
const POSITION_KEYWORDS = new Set(["left", "center", "right", "top", "middle", "bottom"]);
const OVERLAY_TYPES = new Set(["text"]); // image overlays : v2
const MAX_MEDIA = 50;
const MAX_OVERLAYS = 20;
const MIN_FPS = 12;
const MAX_FPS = 60;
const MIN_DURATION_SEC = 0.1;
const MAX_DURATION_SEC = 300;
const MAX_TOTAL_DURATION_SEC = 1800; // 30 min cap
const HEX_COLOR = /^#[0-9A-Fa-f]{6}$/;

function _err(msg) {
  const e = new Error(msg);
  e.status = 400;
  return e;
}

function _isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function _validateColor(v, field) {
  if (typeof v !== "string" || !HEX_COLOR.test(v)) {
    throw _err(`${field}: couleur hex invalide (attendu "#RRGGBB").`);
  }
  return v;
}

function _validateStyle(style) {
  if (!style || typeof style !== "object") return {};
  const out = {};
  if (style.fontname !== undefined) {
    if (typeof style.fontname !== "string" || !/^[A-Za-z0-9 _-]{1,40}$/.test(style.fontname)) {
      throw _err("subtitle.style.fontname: caractères ASCII / espaces uniquement (max 40).");
    }
    out.fontname = style.fontname;
  }
  if (style.fontsize !== undefined) {
    if (!Number.isInteger(style.fontsize) || style.fontsize < 12 || style.fontsize > 200) {
      throw _err("subtitle.style.fontsize: entier dans [12, 200].");
    }
    out.fontsize = style.fontsize;
  }
  for (const f of ["primary_color", "secondary_color", "outline_color", "back_color"]) {
    if (style[f] !== undefined) {
      const arr = style[f];
      if (
        !Array.isArray(arr) || arr.length !== 4 ||
        !arr.every((n) => Number.isInteger(n) && n >= 0 && n <= 255)
      ) {
        throw _err(`subtitle.style.${f}: tableau [r,g,b,a] d'entiers 0-255.`);
      }
      out[f] = arr;
    }
  }
  if (style.alignment !== undefined) {
    if (!Number.isInteger(style.alignment) || style.alignment < 1 || style.alignment > 9) {
      throw _err("subtitle.style.alignment: entier ASS 1-9.");
    }
    out.alignment = style.alignment;
  }
  if (style.margin_v !== undefined) {
    if (!Number.isInteger(style.margin_v) || style.margin_v < 0 || style.margin_v > 2000) {
      throw _err("subtitle.style.margin_v: entier dans [0, 2000].");
    }
    out.margin_v = style.margin_v;
  }
  if (style.outline !== undefined) {
    if (!_isFiniteNumber(style.outline) || style.outline < 0 || style.outline > 20) {
      throw _err("subtitle.style.outline: nombre dans [0, 20].");
    }
    out.outline = style.outline;
  }
  if (style.shadow !== undefined) {
    if (!_isFiniteNumber(style.shadow) || style.shadow < 0 || style.shadow > 20) {
      throw _err("subtitle.style.shadow: nombre dans [0, 20].");
    }
    out.shadow = style.shadow;
  }
  if (style.bold !== undefined) {
    if (typeof style.bold !== "boolean") {
      throw _err("subtitle.style.bold: boolean.");
    }
    out.bold = style.bold;
  }
  if (style.max_words_per_line !== undefined) {
    if (
      !Number.isInteger(style.max_words_per_line) ||
      style.max_words_per_line < 1 || style.max_words_per_line > 20
    ) {
      throw _err("subtitle.style.max_words_per_line: entier dans [1, 20].");
    }
    out.max_words_per_line = style.max_words_per_line;
  }
  return out;
}

function _validateMediaItem(item, index) {
  if (!item || typeof item !== "object") {
    throw _err(`media[${index}]: doit être un objet.`);
  }
  const out = {};
  // type optionnel : si absent, déduit du MIME du fichier uploadé
  if (item.type !== undefined) {
    if (!["image", "video"].includes(item.type)) {
      throw _err(`media[${index}].type: "image" ou "video".`);
    }
    out.type = item.type;
  }
  if (item.duration !== undefined) {
    if (item.duration === "auto") {
      // Sentinelle : la durée sera calculée par le render selon l'audio.
      out.duration = "auto";
    } else if (
      !_isFiniteNumber(item.duration) ||
      item.duration < MIN_DURATION_SEC ||
      item.duration > MAX_DURATION_SEC
    ) {
      throw _err(
        `media[${index}].duration: nombre dans [${MIN_DURATION_SEC}, ${MAX_DURATION_SEC}] secondes, ou la chaîne "auto" pour matcher la durée audio.`
      );
    } else {
      out.duration = item.duration;
    }
  }
  if (item.trim_start !== undefined) {
    if (!_isFiniteNumber(item.trim_start) || item.trim_start < 0 || item.trim_start > MAX_DURATION_SEC) {
      throw _err(`media[${index}].trim_start: nombre dans [0, ${MAX_DURATION_SEC}].`);
    }
    out.trim_start = item.trim_start;
  }
  if (item.trim_end !== undefined) {
    if (!_isFiniteNumber(item.trim_end) || item.trim_end <= 0 || item.trim_end > MAX_DURATION_SEC) {
      throw _err(`media[${index}].trim_end: nombre dans (0, ${MAX_DURATION_SEC}].`);
    }
    out.trim_end = item.trim_end;
  }
  if (out.trim_start !== undefined && out.trim_end !== undefined && out.trim_end <= out.trim_start) {
    throw _err(`media[${index}]: trim_end doit être > trim_start.`);
  }
  if (item.fit !== undefined) {
    if (!FIT_MODES.has(item.fit)) {
      throw _err(`media[${index}].fit: ${[...FIT_MODES].join(" | ")}.`);
    }
    out.fit = item.fit;
  }
  return out;
}

/**
 * Valide ET normalise une config résolue (déjà mergée avec le template).
 * Renvoie la config validée + le format résolu.
 */
function validateConfig(config) {
  if (!config || typeof config !== "object") {
    throw _err("config: doit être un objet JSON.");
  }

  // Format
  const format = resolveFormat(config.format ?? "9:16");

  // FPS
  let fps = config.fps ?? 30;
  if (!Number.isInteger(fps) || fps < MIN_FPS || fps > MAX_FPS) {
    throw _err(`fps: entier dans [${MIN_FPS}, ${MAX_FPS}].`);
  }

  // Background
  const background = _validateColor(config.background ?? "#000000", "background");

  // Subtitle
  const sub = config.subtitle ?? {};
  const subtitleMode = sub.mode ?? "none";
  if (!SUBTITLE_MODES.has(subtitleMode)) {
    throw _err(`subtitle.mode: ${[...SUBTITLE_MODES].join(" | ")}.`);
  }
  const subtitleLanguage = sub.language ?? "fr";
  if (typeof subtitleLanguage !== "string" || !/^[a-z]{2}(-[A-Z]{2})?$/.test(subtitleLanguage)) {
    throw _err(`subtitle.language: code BCP-47 court (ex: "fr", "en", "fr-FR").`);
  }
  const subtitleStyle = _validateStyle(sub.style);

  // Default media
  const defaultMedia = _validateMediaItem(config.default_media ?? {}, "default_media");

  // Media list
  const media = Array.isArray(config.media) ? config.media : [];
  if (media.length === 0) {
    throw _err("media: doit contenir au moins 1 entrée (objet vide accepté pour défauts).");
  }
  if (media.length > MAX_MEDIA) {
    throw _err(`media: max ${MAX_MEDIA} éléments.`);
  }
  const validatedMedia = media.map((m, i) => ({
    ...defaultMedia,
    ..._validateMediaItem(m, i),
  }));

  // Audio handling : que faire si l'audio dépasse la durée des médias ?
  const audio_handling = config.audio_handling ?? "extend";
  if (!AUDIO_HANDLING.has(audio_handling)) {
    throw _err(`audio_handling: ${[...AUDIO_HANDLING].join(" | ")}.`);
  }

  // Transitions
  const transitions = _validateTransitions(config.transitions);

  // Audio mix
  const audio = _validateAudio(config.audio);

  // Text overlays
  const overlays = _validateOverlays(config.overlays);

  // Output
  const output = config.output ?? {};
  const preset = output.preset ?? "medium";
  if (!FFMPEG_PRESETS.has(preset)) {
    throw _err(`output.preset: ${[...FFMPEG_PRESETS].join(" | ")}.`);
  }
  const crf = output.crf ?? 20;
  if (!Number.isInteger(crf) || crf < 0 || crf > 51) {
    throw _err("output.crf: entier dans [0, 51].");
  }

  return {
    format,
    fps,
    background,
    subtitle: {
      mode: subtitleMode,
      language: subtitleLanguage,
      style: subtitleStyle,
    },
    media: validatedMedia,
    audio_handling,
    transitions,
    audio,
    overlays,
    output: { preset, crf },
    // Garde-fou durée totale calculé plus tard une fois les vidéos probées
    limits: { MAX_TOTAL_DURATION_SEC },
  };
}

// ---------------------------------------------------------------
// Sous-validateurs : transitions, audio, overlays
// ---------------------------------------------------------------
function _validateTransitions(t) {
  if (!t) return { type: "none", duration: 0 };
  if (typeof t !== "object") throw _err("transitions: objet attendu.");
  const type = t.type ?? "none";
  if (!XFADE_TYPES.has(type)) {
    throw _err(`transitions.type: l'une de ${[...XFADE_TYPES].join(" | ")}.`);
  }
  if (type === "none") return { type: "none", duration: 0 };
  const duration = t.duration ?? 0.3;
  if (!_isFiniteNumber(duration) || duration < 0.1 || duration > 2.0) {
    throw _err("transitions.duration: nombre dans [0.1, 2.0] secondes.");
  }
  return { type, duration };
}

function _validateAudio(a) {
  if (!a) return { music_volume: 0.3, voice_volume: 1.0, ducking: false };
  if (typeof a !== "object") throw _err("audio: objet attendu.");
  const out = { music_volume: 0.3, voice_volume: 1.0, ducking: false };
  if (a.music_volume !== undefined) {
    if (!_isFiniteNumber(a.music_volume) || a.music_volume < 0 || a.music_volume > 2) {
      throw _err("audio.music_volume: nombre dans [0, 2].");
    }
    out.music_volume = a.music_volume;
  }
  if (a.voice_volume !== undefined) {
    if (!_isFiniteNumber(a.voice_volume) || a.voice_volume < 0 || a.voice_volume > 2) {
      throw _err("audio.voice_volume: nombre dans [0, 2].");
    }
    out.voice_volume = a.voice_volume;
  }
  if (a.ducking !== undefined) {
    if (typeof a.ducking !== "boolean") {
      throw _err("audio.ducking: boolean.");
    }
    out.ducking = a.ducking;
  }
  return out;
}

function _validatePosition(v, field) {
  // Accepte un mot-clé ("center", "top"...) ou un entier de pixels.
  if (typeof v === "string") {
    if (!POSITION_KEYWORDS.has(v)) {
      throw _err(`${field}: ${[...POSITION_KEYWORDS].join(" | ")} ou entier (pixels).`);
    }
    return v;
  }
  if (Number.isInteger(v) && v >= 0 && v <= 8000) return v;
  throw _err(`${field}: mot-clé valide ou entier de pixels [0, 8000].`);
}

function _validateOverlays(list) {
  if (!list) return [];
  if (!Array.isArray(list)) throw _err("overlays: tableau attendu.");
  if (list.length > MAX_OVERLAYS) {
    throw _err(`overlays: max ${MAX_OVERLAYS} éléments.`);
  }
  return list.map((o, i) => _validateOverlay(o, i));
}

// drawtext utilise `text='...'`. Caractères à neutraliser pour ne pas
// casser le filter graph OU ouvrir une injection : `\`, `'`, `:`, `%`, `,`.
const SAFE_OVERLAY_TEXT = /^[^\x00-\x1F]{0,500}$/; // pas de contrôles, max 500 chars

function _validateOverlay(o, i) {
  if (!o || typeof o !== "object") {
    throw _err(`overlays[${i}]: objet attendu.`);
  }
  if (!OVERLAY_TYPES.has(o.type)) {
    throw _err(`overlays[${i}].type: ${[...OVERLAY_TYPES].join(" | ")}.`);
  }
  if (typeof o.text !== "string" || !SAFE_OVERLAY_TEXT.test(o.text)) {
    throw _err(`overlays[${i}].text: chaîne sans caractères de contrôle, max 500.`);
  }
  const out = { type: o.type, text: o.text };

  if (o.start !== undefined) {
    if (!_isFiniteNumber(o.start) || o.start < 0 || o.start > MAX_DURATION_SEC) {
      throw _err(`overlays[${i}].start: nombre dans [0, ${MAX_DURATION_SEC}].`);
    }
    out.start = o.start;
  }
  if (o.end !== undefined && o.end !== null) {
    if (!_isFiniteNumber(o.end) || o.end <= 0 || o.end > MAX_DURATION_SEC) {
      throw _err(`overlays[${i}].end: nombre dans (0, ${MAX_DURATION_SEC}].`);
    }
    out.end = o.end;
  }
  if (out.start !== undefined && out.end !== undefined && out.end <= out.start) {
    throw _err(`overlays[${i}]: end doit être > start.`);
  }
  out.x = _validatePosition(o.x ?? "center", `overlays[${i}].x`);
  out.y = _validatePosition(o.y ?? "top", `overlays[${i}].y`);

  if (o.fontsize !== undefined) {
    if (!Number.isInteger(o.fontsize) || o.fontsize < 8 || o.fontsize > 300) {
      throw _err(`overlays[${i}].fontsize: entier dans [8, 300].`);
    }
    out.fontsize = o.fontsize;
  } else {
    out.fontsize = 48;
  }
  if (o.color !== undefined) {
    out.color = _validateColor(o.color, `overlays[${i}].color`);
  } else {
    out.color = "#FFFFFF";
  }
  if (o.bg_color !== undefined && o.bg_color !== null) {
    out.bg_color = _validateColor(o.bg_color, `overlays[${i}].bg_color`);
  }
  if (o.bg_opacity !== undefined) {
    if (!_isFiniteNumber(o.bg_opacity) || o.bg_opacity < 0 || o.bg_opacity > 1) {
      throw _err(`overlays[${i}].bg_opacity: nombre dans [0, 1].`);
    }
    out.bg_opacity = o.bg_opacity;
  } else {
    out.bg_opacity = 0.6;
  }
  return out;
}

module.exports = { validateConfig };
