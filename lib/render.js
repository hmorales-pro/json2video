/**
 * Moteur de composition vidéo basé sur FFmpeg.
 *
 * Pipeline (modulaire) :
 *   1) Préparer chaque média → chaîne vidéo normalisée [v_i]
 *   2) Assembler [v_0..v_n] via concat OU xfade selon transitions.type
 *   3) Appliquer les sous-titres (subtitles=) si demandé
 *   4) Appliquer les overlays texte (drawtext) si demandés
 *   5) Mixer audio : voice + music (amix, optionnellement sidechaincompress pour ducking)
 *   6) Encodage final (libx264 + AAC + faststart + yuv420p)
 *
 * Sécurité : tous les chemins viennent d'UUIDs serveur. Les textes
 * d'overlays passent par drawtext avec `textfile=` (lu en fichier),
 * jamais interpolé brut dans la commande. spawn (pas de shell).
 */
const { spawn } = require("child_process");
const fs = require("fs/promises");
const path = require("path");
const { v4: uuidv4 } = require("uuid");

const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const FFMPEG_BIN = process.env.FFMPEG_BIN || "ffmpeg";
const FFPROBE_BIN = process.env.FFPROBE_BIN || "ffprobe";
const FONT_PATH = process.env.OVERLAY_FONT_PATH
  || "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf";

// ---------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------
function runCommand(
  bin,
  args,
  {
    timeoutMs = 10 * 60 * 1000,
    logTag = bin,
    heartbeatEverySec = 0, // 0 = pas de heartbeat
  } = {}
) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    if (heartbeatEverySec > 0) {
      console.log(`[${logTag}] starting...`);
    }
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    let stdout = "";
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);

    // Heartbeat : log toutes les N secondes pour voir que ça tourne
    // (FFmpeg peut prendre plusieurs minutes sans rien afficher).
    let heartbeatTimer = null;
    if (heartbeatEverySec > 0) {
      heartbeatTimer = setInterval(() => {
        const elapsed = Math.round((Date.now() - startTime) / 1000);
        console.log(`[${logTag}] still running (${elapsed}s elapsed)...`);
      }, heartbeatEverySec * 1000);
      heartbeatTimer.unref();
    }

    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.stdout.on("data", (d) => { stdout += d.toString(); });

    const cleanup = () => {
      clearTimeout(timer);
      if (heartbeatTimer) clearInterval(heartbeatTimer);
    };

    child.on("error", (err) => {
      cleanup();
      reject(err);
    });
    child.on("close", (code) => {
      cleanup();
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      if (code === 0) {
        if (heartbeatEverySec > 0) console.log(`[${logTag}] done in ${elapsed}s`);
        return resolve({ stdout, stderr });
      }
      const err = new Error(`${logTag} exited with code ${code} after ${elapsed}s\n${stderr.slice(-2000)}`);
      err.stderr = stderr;
      reject(err);
    });
  });
}

async function probeMedia(filePath) {
  const args = [
    "-v", "error",
    "-print_format", "json",
    "-show_streams",
    "-show_format",
    filePath,
  ];
  const { stdout } = await runCommand(FFPROBE_BIN, args, { logTag: "ffprobe", timeoutMs: 30_000 });
  const data = JSON.parse(stdout || "{}");
  const streams = data.streams || [];
  const v = streams.find((s) => s.codec_type === "video");
  const a = streams.find((s) => s.codec_type === "audio");
  return {
    width: v?.width || 0,
    height: v?.height || 0,
    duration_sec: parseFloat(data.format?.duration ?? "0") || 0,
    has_video: !!v,
    has_audio: !!a,
  };
}

// ---------------------------------------------------------------
// Position helpers (drawtext)
// ---------------------------------------------------------------
function _positionX(x) {
  if (typeof x === "number") return String(x);
  switch (x) {
    case "left":   return "20";
    case "center": return "(w-text_w)/2";
    case "right":  return "w-text_w-20";
    default:       return "(w-text_w)/2";
  }
}
function _positionY(y) {
  if (typeof y === "number") return String(y);
  switch (y) {
    case "top":    return "20";
    case "middle": return "(h-text_h)/2";
    case "bottom": return "h-text_h-20";
    default:       return "20";
  }
}

// drawtext exige un échappement spécifique pour la valeur `boxcolor`.
function _hexToFFColor(hex, opacity = null) {
  // "#RRGGBB" → "0xRRGGBB" + optional @0.5 for opacity (drawtext convention)
  const c = "0x" + hex.replace("#", "").toUpperCase();
  if (opacity !== null) return `${c}@${opacity.toFixed(2)}`;
  return c;
}

// ---------------------------------------------------------------
// Construction des chaînes vidéo
// ---------------------------------------------------------------
function _videoFilterForItem(item, format, fps, mediaIndex) {
  const { width, height } = format;
  const inLabel = `${mediaIndex}:v`;

  const fitFilters = (() => {
    switch (item.fit || "cover") {
      case "cover":
        return [
          `scale=${width}:${height}:force_original_aspect_ratio=increase`,
          `crop=${width}:${height}`,
        ];
      case "contain":
        return [
          `scale=${width}:${height}:force_original_aspect_ratio=decrease`,
          `pad=${width}:${height}:(${width}-iw)/2:(${height}-ih)/2:color=black`,
        ];
      case "stretch":
        return [`scale=${width}:${height}`];
      default:
        return [`scale=${width}:${height}`];
    }
  })();

  const trimFilter = (() => {
    if (item.__sourceType === "image") return [];
    // En mode loop, l'input -t suffit ; le trim filter recouperait au-delà
    // de la durée native de la source et casserait la boucle.
    if (item.__loop_extend > 0) return [];
    const start = item.trim_start;
    const end = item.trim_end;
    if (start === undefined && end === undefined) return [];
    const parts = [];
    if (start !== undefined && end !== undefined) parts.push(`trim=start=${start}:end=${end}`);
    else if (start !== undefined)                parts.push(`trim=start=${start}`);
    else if (end !== undefined)                  parts.push(`trim=end=${end}`);
    parts.push("setpts=PTS-STARTPTS");
    return parts;
  })();

  const freezeFilter =
    item.__sourceType === "video" && item.__freeze_extend > 0
      ? [`tpad=stop_mode=clone:stop_duration=${item.__freeze_extend.toFixed(3)}`]
      : [];

  const allFilters = [
    ...fitFilters,
    `fps=${fps}`,
    "setsar=1",
    ...trimFilter,
    ...freezeFilter,
    "format=yuv420p",
  ];

  return `[${inLabel}]${allFilters.join(",")}[v${mediaIndex}]`;
}

// ---------------------------------------------------------------
// Assemblage : concat OU xfade
// ---------------------------------------------------------------
function _buildConcatChain(items, outLabel) {
  const inputs = items.map((_, i) => `[v${i}]`).join("");
  return `${inputs}concat=n=${items.length}:v=1:a=0${outLabel}`;
}

function _buildXfadeChain(items, transitions, outLabel) {
  const td = transitions.duration;
  if (items.length === 1) {
    // Une seule chaîne : on renomme simplement [v0] → outLabel via null filter
    return `[v0]null${outLabel}`;
  }
  const lines = [];
  // running = somme des durées des médias déjà fusionnés (avant cette transition)
  let running = items[0].__duration_sec;
  let prevLabel = "v0";
  for (let i = 1; i < items.length; i++) {
    const offset = running - td * i; // décale chaque fois de td pour absorber la transition
    const nextLabel = i === items.length - 1 ? outLabel.slice(1, -1) : `vt${i}`;
    const labelTag = i === items.length - 1 ? outLabel : `[${nextLabel}]`;
    lines.push(
      `[${prevLabel}][v${i}]xfade=transition=${transitions.type}:duration=${td}:offset=${offset.toFixed(3)}${labelTag}`
    );
    running += items[i].__duration_sec;
    prevLabel = nextLabel;
  }
  return lines.join(";");
}

// ---------------------------------------------------------------
// Filtres post-assemblage : subtitles + overlays
// ---------------------------------------------------------------
function _subtitleFilter(inLabel, outLabel, subtitlesPath) {
  // FFmpeg ≥ 6 refuse la forme positionnelle `subtitles=/path/file.ass`
  // (parsée comme une liste d'options key=value : "No option name near ...").
  // Forme universelle : `subtitles=filename=/path/file.ass`.
  //
  // Le path est mis en absolu et les caractères spéciaux sont échappés.
  const path = require("path");
  const abs = path.resolve(subtitlesPath);
  const escaped = abs
    .replace(/\\/g, "\\\\")
    .replace(/:/g, "\\:")
    .replace(/,/g, "\\,")
    .replace(/'/g, "\\\\'");
  return `${inLabel}subtitles=filename=${escaped}${outLabel}`;
}

function _overlayFilters(inLabel, outLabel, overlays, overlayTextFiles) {
  if (!overlays.length) return null;
  // On chaîne plusieurs drawtext consécutifs sur un même label.
  const drawTexts = overlays.map((o, i) => {
    const parts = [
      `fontfile='${FONT_PATH}'`,
      `textfile='${overlayTextFiles[i]}'`, // texte lu depuis un fichier (pas d'injection)
      `fontsize=${o.fontsize}`,
      `fontcolor=${_hexToFFColor(o.color)}`,
      `x=${_positionX(o.x)}`,
      `y=${_positionY(o.y)}`,
    ];
    if (o.bg_color) {
      parts.push("box=1");
      parts.push(`boxcolor=${_hexToFFColor(o.bg_color, o.bg_opacity)}`);
      parts.push("boxborderw=10");
    }
    // Conditionner par start/end via enable=
    const enable = [];
    if (o.start !== undefined) enable.push(`gte(t\\,${o.start})`);
    if (o.end !== undefined) enable.push(`lt(t\\,${o.end})`);
    if (enable.length) parts.push(`enable='${enable.join("*")}'`);
    return parts.join(":");
  });
  // Empile les drawtext via une chaîne de filtres
  return `${inLabel}${drawTexts.map((d) => `drawtext=${d}`).join(",")}${outLabel}`;
}

// ---------------------------------------------------------------
// Mix audio (voice + music)
// ---------------------------------------------------------------
function _buildAudioFilter(audioInputs, audioCfg) {
  // audioInputs = [{label, role:"voice"|"music"}]
  if (audioInputs.length === 0) return null;
  const voice = audioInputs.find((a) => a.role === "voice");
  const music = audioInputs.find((a) => a.role === "music");

  if (voice && !music) {
    // Juste la voix, on peut quand même appliquer un gain.
    return `[${voice.label}]volume=${audioCfg.voice_volume}[aout]`;
  }
  if (music && !voice) {
    return `[${music.label}]volume=${audioCfg.music_volume}[aout]`;
  }
  if (!voice && !music) return null;

  if (audioCfg.ducking) {
    // Sidechain compress : la musique baisse quand la voix est présente.
    // [music] passe par sidechaincompress avec [voice] comme sidechain.
    return (
      `[${voice.label}]volume=${audioCfg.voice_volume},asplit=2[voice_main][voice_sc];` +
      `[${music.label}]volume=${audioCfg.music_volume}[music_pre];` +
      `[music_pre][voice_sc]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=400[music_ducked];` +
      `[voice_main][music_ducked]amix=inputs=2:normalize=0:duration=first[aout]`
    );
  }
  // amix simple avec pondération
  return (
    `[${voice.label}]volume=${audioCfg.voice_volume}[voice_pre];` +
    `[${music.label}]volume=${audioCfg.music_volume}[music_pre];` +
    `[voice_pre][music_pre]amix=inputs=2:normalize=0:duration=first[aout]`
  );
}

// ---------------------------------------------------------------
// API publique : buildFilterComplex (utilisée par dry-run + tests)
// ---------------------------------------------------------------
function buildFilterComplex(items, format, fps, subtitlesPath, opts = {}) {
  const transitions = opts.transitions || { type: "none", duration: 0 };
  const overlays = opts.overlays || [];
  const overlayTextFiles = opts.overlayTextFiles || [];
  const audioInputs = opts.audioInputs || [];
  const audioCfg = opts.audioCfg || { voice_volume: 1.0, music_volume: 0.3, ducking: false };

  const chains = items.map((item, i) => _videoFilterForItem(item, format, fps, i));

  // Détermine la liste des post-filtres à appliquer après concat/xfade.
  const postSteps = [];
  if (subtitlesPath) postSteps.push("sub");
  if (overlays.length) postSteps.push("ovr");

  // Si aucun post-filtre, l'assemblage écrit DIRECTEMENT dans [vout].
  // Sinon il écrit dans [vasm] et on chaîne ensuite jusqu'à [vout].
  const useXfade = transitions.type !== "none" && items.length > 1;
  const assembleOut = postSteps.length ? "[vasm]" : "[vout]";
  if (useXfade) {
    chains.push(_buildXfadeChain(items, transitions, assembleOut));
  } else {
    chains.push(_buildConcatChain(items, assembleOut));
  }

  let currentLabel = assembleOut;
  for (let i = 0; i < postSteps.length; i++) {
    const isLast = i === postSteps.length - 1;
    const out = isLast ? "[vout]" : `[v${postSteps[i]}]`;
    if (postSteps[i] === "sub") {
      chains.push(_subtitleFilter(currentLabel, out, subtitlesPath));
    } else {
      chains.push(_overlayFilters(currentLabel, out, overlays, overlayTextFiles));
    }
    currentLabel = out;
  }

  const audioChain = _buildAudioFilter(audioInputs, audioCfg);
  if (audioChain) chains.push(audioChain);

  return chains.join(";");
}

// ---------------------------------------------------------------
// Préparation des inputs ffmpeg
// ---------------------------------------------------------------
async function _prepareMediaItems(mediaFiles, mediaConfigs, audioFile, transitionsOverlapSec = 0) {
  // 1ère passe : classification et durées FIXES.
  //   - Vidéos : durée native (trim appliqué)
  //   - Images avec durée numérique : durée explicite
  //   - Images avec "auto" ou sans durée : compteront pour la répartition
  const tmp = [];
  let fixedTotalSec = 0;
  let autoCount = 0;

  for (let i = 0; i < mediaFiles.length; i++) {
    const file = mediaFiles[i];
    const cfg = mediaConfigs[i] || {};
    const isImage = file.mimetype?.startsWith("image/") || cfg.type === "image";
    const isVideo = file.mimetype?.startsWith("video/") || cfg.type === "video";
    if (!isImage && !isVideo) {
      throw Object.assign(
        new Error(`media[${i}]: type non supporté (${file.mimetype}).`),
        { status: 400 }
      );
    }
    if (isImage) {
      const isAuto = cfg.duration === "auto" || cfg.duration === undefined;
      if (isAuto) {
        autoCount++;
        tmp.push({ file, cfg, sourceType: "image", duration: null });
      } else {
        fixedTotalSec += cfg.duration;
        tmp.push({ file, cfg, sourceType: "image", duration: cfg.duration });
      }
    } else {
      const probe = await probeMedia(file.path);
      const start = cfg.trim_start ?? 0;
      const end = cfg.trim_end ?? probe.duration_sec;
      const dur = Math.max(0.1, end - start);
      fixedTotalSec += dur;
      tmp.push({ file, cfg, sourceType: "video", duration: dur });
    }
  }

  // Probe audio.
  let audioDurationSec = null;
  if (audioFile) {
    const probe = await probeMedia(audioFile.path);
    audioDurationSec = probe.duration_sec;
  }

  // Calcul de la durée auto.
  // Avec concat : total_video = somme(durations) = audio.
  // Avec xfade : total_video = somme(durations) - (N-1)*td = audio
  //              donc somme(durations) = audio + transitionsOverlapSec
  let autoDuration = 3.0; // fallback si pas d'audio
  if (autoCount > 0 && audioDurationSec !== null) {
    const target = audioDurationSec + transitionsOverlapSec;
    const remaining = target - fixedTotalSec;
    if (remaining < 0.3 * autoCount) {
      // Audio trop court pour absorber : on tombe à 0.3s mini par image
      autoDuration = 0.3;
    } else {
      autoDuration = remaining / autoCount;
    }
  }

  // 2ème passe : matérialisation.
  const items = tmp.map((t) => ({
    ...t.cfg,
    __sourceType: t.sourceType,
    __path: path.resolve(t.file.path),
    __duration_sec: t.duration === null ? autoDuration : t.duration,
    __auto: t.duration === null,
    __freeze_extend: 0,
    __loop_extend: 0,
  }));

  return { items, audioDurationSec, autoDuration, autoCount };
}

/**
 * Si l'audio dépasse la somme des durées vidéo, on étend le DERNIER item
 * pour combler le déficit selon `mode` :
 *
 *   - image  (toujours) → on bump __duration_sec, l'image étant en `-loop 1`
 *   - vidéo + mode "extend" → __freeze_extend (tpad freeze sur la dernière frame)
 *   - vidéo + mode "loop"   → __loop_extend (`-stream_loop -1` côté input)
 *
 * Renvoie le nombre de secondes ajoutées.
 */
function _extendLastIfAudioOverflow(items, audioDurationSec, transitionsOverlapSec, mode) {
  if (!items.length || audioDurationSec == null) return { extended: 0, kind: null };
  const videoTotal = items.reduce((a, b) => a + b.__duration_sec, 0) - transitionsOverlapSec;
  const deficit = audioDurationSec - videoTotal;
  if (deficit <= 0.05) return { extended: 0, kind: null };

  const last = items[items.length - 1];
  if (last.__sourceType === "image") {
    last.__duration_sec += deficit;
    return { extended: deficit, kind: "image_extend" };
  }
  if (mode === "loop") {
    last.__loop_extend = deficit;
    return { extended: deficit, kind: "video_loop" };
  }
  // default: "extend" → freeze
  last.__freeze_extend = deficit;
  return { extended: deficit, kind: "video_freeze" };
}

const LOOP_ALL_MAX_K = 50; // garde-fou : limite le nombre de répétitions

/**
 * Mode "loop_all" : duplique la totalité de la séquence d'items autant de
 * fois que nécessaire pour couvrir l'audio. -shortest coupera ensuite pile
 * à la fin de l'audio.
 *
 * Mutation in-place du tableau `items`.
 */
function _applyLoopAll(items, audioDurationSec, transitionsOverlapSec) {
  if (!items.length) return { K: 1 };
  const baseTotal =
    items.reduce((a, b) => a + b.__duration_sec, 0) - transitionsOverlapSec;
  if (baseTotal <= 0.05) return { K: 1 };
  if (audioDurationSec <= baseTotal + 0.05) return { K: 1 };

  let K = Math.ceil(audioDurationSec / baseTotal);
  if (K > LOOP_ALL_MAX_K) {
    console.warn(
      `[render] loop_all : K=${K} > limite ${LOOP_ALL_MAX_K}. ` +
      `Cappé à ${LOOP_ALL_MAX_K} (séquence + audio coupé).`
    );
    K = LOOP_ALL_MAX_K;
  }
  if (K <= 1) return { K: 1 };

  const original = items.slice();
  for (let c = 1; c < K; c++) {
    for (const it of original) {
      // shallow copy : __path partagé est OK, ffmpeg ouvre N descripteurs
      items.push({ ...it });
    }
  }
  console.log(
    `[render] loop_all : séquence répétée ${K}× ` +
    `(${original.length} items × ${K} = ${items.length} inputs) ` +
    `pour couvrir ${audioDurationSec.toFixed(2)}s d'audio.`
  );
  return { K };
}

function _ffmpegInputArgsForItem(item) {
  if (item.__sourceType === "image") {
    return ["-loop", "1", "-t", String(item.__duration_sec.toFixed(3))];
  }
  const args = [];
  // En mode loop, on demande à ffmpeg de boucler l'input à l'infini ;
  // c'est `-t` qui borne la durée totale visible.
  if (item.__loop_extend > 0) {
    args.push("-stream_loop", "-1");
  }
  if (item.trim_start !== undefined) args.push("-ss", String(item.trim_start));

  // Durée totale visible de l'item après extension éventuelle.
  // - loop : __duration_sec + __loop_extend (couverture par -t)
  // - freeze : __duration_sec (le tpad ajoutera __freeze_extend en filtre)
  // - normal : __duration_sec
  const targetDur =
    item.__duration_sec +
    (item.__loop_extend > 0 ? item.__loop_extend : 0);
  args.push("-t", String(targetDur.toFixed(3)));
  return args;
}

// ---------------------------------------------------------------
// Génération sous-titres
// ---------------------------------------------------------------
async function _generateSubtitles({ mode, language, style, audioFile, tmpDir, transcribeFns }) {
  if (mode === "none" || !audioFile) return null;
  const styleJsonPath = path.resolve(path.join(tmpDir, `${uuidv4()}_style.json`));
  await fs.writeFile(styleJsonPath, JSON.stringify(style), "utf8");
  const assPath = path.resolve(path.join(tmpDir, `${uuidv4()}.ass`));

  if (mode === "karaoke") {
    const whisperJson = await transcribeFns.wordLevel(audioFile, language);
    const jsonPath = path.join(tmpDir, `${uuidv4()}_words.json`);
    await fs.writeFile(jsonPath, JSON.stringify(whisperJson), "utf8");
    await runCommand(
      PYTHON_BIN,
      [path.resolve(__dirname, "..", "words_to_ass.py"), jsonPath, assPath, "--style", styleJsonPath],
      { logTag: "words_to_ass", timeoutMs: 60_000 }
    );
    return { assPath, cleanup: [jsonPath, styleJsonPath] };
  }
  const srt = await transcribeFns.srt(audioFile, language);
  const srtPath = path.join(tmpDir, `${uuidv4()}.srt`);
  await fs.writeFile(srtPath, srt, "utf8");
  await runCommand(
    PYTHON_BIN,
    [path.resolve(__dirname, "..", "convert_srt_to_ass.py"), srtPath, assPath, "--style", styleJsonPath],
    { logTag: "convert_srt_to_ass", timeoutMs: 60_000 }
  );
  return { assPath, cleanup: [srtPath, styleJsonPath] };
}

// ---------------------------------------------------------------
// API : render
// ---------------------------------------------------------------
/**
 * @param {object} params
 * @param {object} params.config       Config validée (cf. lib/validate.js).
 * @param {Array}  params.mediaFiles   Fichiers multer dans l'ordre.
 * @param {object} [params.audioFile]  Fichier voix-off / audio principal.
 * @param {object} [params.musicFile]  Fichier musique de fond.
 * @param {object} params.transcribeFns  { srt, wordLevel }
 * @param {string} params.outputPath
 * @param {string} params.tmpDir
 */
async function render({
  config,
  mediaFiles,
  audioFile,
  musicFile,
  transcribeFns,
  outputPath,
  tmpDir = "uploads",
}) {
  if (!Array.isArray(mediaFiles) || mediaFiles.length === 0) {
    throw Object.assign(new Error("Aucun média fourni."), { status: 400 });
  }
  if (mediaFiles.length !== config.media.length) {
    throw Object.assign(
      new Error(`Mismatch: ${mediaFiles.length} fichiers vs ${config.media.length} entrées config.media.`),
      { status: 400 }
    );
  }
  // Avec xfade, il faut au moins 2 médias pour une transition utile.
  // Sinon on ignore silencieusement.
  const useXfade =
    config.transitions.type !== "none" && mediaFiles.length > 1;

  // Chaque transition xfade rogne (td) secondes du total ; on doit le
  // compenser dans le calcul de la durée auto pour que la vidéo finale
  // matche bien la durée audio.
  const transitionsOverlap = useXfade
    ? (mediaFiles.length - 1) * config.transitions.duration
    : 0;

  const { items, audioDurationSec, autoDuration, autoCount } = await _prepareMediaItems(
    mediaFiles, config.media, audioFile, transitionsOverlap
  );

  if (autoCount > 0) {
    console.log(
      `[render] Durée auto: ${autoDuration.toFixed(2)}s par image ` +
      `× ${autoCount} image(s) = ${(autoDuration * autoCount).toFixed(2)}s ` +
      `(audio: ${audioDurationSec?.toFixed(2) ?? "n/a"}s)`
    );
  }

  // Si l'audio dépasse la durée des médias et qu'on n'est pas en "cut" :
  //   - extend   → freeze frame du dernier média
  //   - loop     → boucle uniquement du dernier média
  //   - loop_all → toute la séquence est répétée jusqu'à couvrir l'audio
  if (config.audio_handling !== "cut" && audioDurationSec != null) {
    if (config.audio_handling === "loop_all") {
      _applyLoopAll(items, audioDurationSec, transitionsOverlap);
    } else {
      const { extended, kind } = _extendLastIfAudioOverflow(
        items, audioDurationSec, transitionsOverlap, config.audio_handling
      );
      if (extended > 0) {
        const human = {
          image_extend: "durée d'image allongée",
          video_freeze: "freeze frame (tpad)",
          video_loop: "loop vidéo (stream_loop)",
        }[kind] || kind;
        console.log(
          `[render] Audio plus long que les médias (+${extended.toFixed(2)}s). ` +
          `Extension du dernier média via ${human}.`
        );
      }
    }
  }

  // Durée totale tenant compte des transitions
  let totalDurationSec = items.reduce((a, b) => a + b.__duration_sec, 0);
  if (useXfade) {
    totalDurationSec -= (items.length - 1) * config.transitions.duration;
  }
  if (totalDurationSec > config.limits.MAX_TOTAL_DURATION_SEC) {
    throw Object.assign(
      new Error(`Durée totale ${totalDurationSec.toFixed(1)}s > limite.`),
      { status: 400 }
    );
  }

  // Sous-titres
  const tempPaths = [];
  let subtitlesPath = null;
  if (config.subtitle.mode !== "none" && audioFile) {
    const styleMerged = {
      // PlayResX/Y indispensables — sans eux libass utilise 384x288 par
      // défaut et SCALE tout (fontsize, margin, outline) par le ratio
      // entre la vidéo et ce default → captions x6 trop grosses.
      play_res_x: config.format.width,
      play_res_y: config.format.height,
      fontsize: config.format.default_fontsize,
      margin_v: config.format.default_margin_v,
      alignment: config.format.alignment,
      ...config.subtitle.style,
    };
    const sub = await _generateSubtitles({
      mode: config.subtitle.mode,
      language: config.subtitle.language,
      style: styleMerged,
      audioFile, tmpDir, transcribeFns,
    });
    if (sub) {
      subtitlesPath = sub.assPath;
      tempPaths.push(sub.assPath, ...sub.cleanup);
    }
  }

  // Écrire les textes d'overlay dans des fichiers (drawtext textfile=)
  const overlayTextFiles = [];
  for (let i = 0; i < config.overlays.length; i++) {
    const txtPath = path.join(tmpDir, `${uuidv4()}_overlay_${i}.txt`);
    await fs.writeFile(txtPath, config.overlays[i].text, "utf8");
    overlayTextFiles.push(txtPath);
    tempPaths.push(txtPath);
  }

  // Construction des inputs ffmpeg
  const args = ["-hide_banner", "-y"];
  for (const item of items) {
    args.push(..._ffmpegInputArgsForItem(item));
    args.push("-i", item.__path);
  }
  const audioInputs = [];
  if (audioFile) {
    args.push("-i", path.resolve(audioFile.path));
    audioInputs.push({ label: `${items.length}:a:0`, role: "voice" });
  }
  if (musicFile) {
    args.push("-i", path.resolve(musicFile.path));
    audioInputs.push({
      label: `${items.length + (audioFile ? 1 : 0)}:a:0`,
      role: "music",
    });
  }

  const filterComplex = buildFilterComplex(items, config.format, config.fps, subtitlesPath, {
    transitions: config.transitions,
    overlays: config.overlays,
    overlayTextFiles,
    audioInputs,
    audioCfg: config.audio,
  });

  args.push("-filter_complex", filterComplex);
  args.push("-map", "[vout]");
  if (audioInputs.length) args.push("-map", "[aout]");

  args.push(
    "-c:v", "libx264",
    "-preset", config.output.preset,
    "-crf", String(config.output.crf),
    "-pix_fmt", "yuv420p",
    "-r", String(config.fps),
    "-movflags", "+faststart"
  );
  if (audioInputs.length) args.push("-c:a", "aac", "-b:a", "192k");

  // -shortest ne fonctionne pas bien avec le filtre concat (ffmpeg regarde
  // les INPUTS individuels, pas la sortie du concat). En mode loop_all on
  // borne donc explicitement la durée de sortie à l'audio pour couper net.
  if (config.audio_handling === "loop_all" && audioDurationSec != null) {
    args.push("-t", audioDurationSec.toFixed(3));
  }
  args.push("-shortest", outputPath);

  console.log(
    `[render] FFmpeg invoqué : ${items.length} média(s), ` +
    `${audioInputs.length} input(s) audio, ` +
    `format ${config.format.width}x${config.format.height}@${config.fps}fps, ` +
    `subs=${config.subtitle.mode}, overlays=${config.overlays.length}, ` +
    `transitions=${config.transitions.type}.`
  );
  await runCommand(FFMPEG_BIN, args, {
    logTag: "ffmpeg",
    timeoutMs: 30 * 60 * 1000,
    heartbeatEverySec: 10,
  });

  for (const p of tempPaths) fs.unlink(p).catch(() => {});

  return {
    outputPath,
    ffmpegCmd: args,
    totalDurationSec: audioDurationSec
      ? Math.min(totalDurationSec, audioDurationSec)
      : totalDurationSec,
  };
}

/**
 * Concatène plusieurs fichiers audio en un seul (AAC).
 * Utile quand le client envoie plusieurs morceaux de voix-off (TTS par
 * paragraphe par ex.) qu'on veut jouer à la suite.
 *
 * Utilise le filtre `concat=a=1` qui gère gracieusement les fichiers de
 * codecs / sample rates / canaux différents (re-encode en AAC stéréo 44.1 kHz).
 *
 * @param {Array<{path:string}>} audioInputs  Fichiers à concaténer (ordre conservé)
 * @param {string} outputPath  Chemin du fichier audio résultant
 * @returns {Promise<string>}  outputPath (pour chaînage)
 */
async function concatAudioFiles(audioInputs, outputPath) {
  if (!audioInputs.length) throw new Error("concatAudioFiles: aucun fichier audio");
  if (audioInputs.length === 1) {
    // Un seul fichier : pas besoin de FFmpeg, on retourne le chemin tel quel.
    return audioInputs[0].path;
  }
  const args = ["-hide_banner", "-y"];
  for (const f of audioInputs) args.push("-i", path.resolve(f.path));
  const filterInputs = audioInputs.map((_, i) => `[${i}:a:0]`).join("");
  const filter = `${filterInputs}concat=n=${audioInputs.length}:v=0:a=1[aout]`;
  args.push(
    "-filter_complex", filter,
    "-map", "[aout]",
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "44100",
    "-ac", "2",
    outputPath
  );
  await runCommand(FFMPEG_BIN, args, {
    logTag: "ffmpeg-concat-audio",
    timeoutMs: 5 * 60 * 1000,
    heartbeatEverySec: 10,
  });
  return outputPath;
}

module.exports = {
  render,
  probeMedia,
  buildFilterComplex,
  runCommand,
  concatAudioFiles,
};
