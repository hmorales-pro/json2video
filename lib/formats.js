/**
 * Presets de formats vidéo standards.
 *
 * Chaque preset définit :
 *   - width / height : résolution cible
 *   - default_fontsize / default_margin_v : valeurs ASS par défaut adaptées
 *     au format (un Reel a besoin d'une fonte plus grosse qu'une vidéo
 *     YouTube et de marges plus généreuses).
 *   - alignment : 1=bas-gauche, 2=bas-centre, 5=milieu-gauche, etc. (codes ASS).
 *
 * Le champ format de la config peut être :
 *   - un slug ("9:16", "16:9", "1:1", "4:5", "21:9")
 *   - un objet { width, height } (dimensions custom)
 */

const PRESETS = Object.freeze({
  "9:16": {
    label: "Reels / TikTok / Shorts",
    width: 1080,
    height: 1920,
    default_fontsize: 64,
    default_margin_v: 220,
    alignment: 2, // bas-centre
  },
  "16:9": {
    label: "YouTube / Web",
    width: 1920,
    height: 1080,
    default_fontsize: 48,
    default_margin_v: 80,
    alignment: 2,
  },
  "1:1": {
    label: "Instagram square",
    width: 1080,
    height: 1080,
    default_fontsize: 56,
    default_margin_v: 80,
    alignment: 2,
  },
  "4:5": {
    label: "Instagram portrait",
    width: 1080,
    height: 1350,
    default_fontsize: 60,
    default_margin_v: 120,
    alignment: 2,
  },
  "21:9": {
    label: "Cinéma / Ultra-wide",
    width: 2560,
    height: 1080,
    default_fontsize: 52,
    default_margin_v: 90,
    alignment: 2,
  },
});

const MIN_DIM = 16;
const MAX_DIM = 4096;

/**
 * Résout un format (slug ou objet) en preset normalisé.
 * Lève une erreur si le format est invalide.
 */
function resolveFormat(format) {
  if (typeof format === "string") {
    const preset = PRESETS[format];
    if (!preset) {
      throw new Error(
        `Format inconnu: "${format}". Valeurs autorisées: ${Object.keys(PRESETS).join(", ")} ou {width,height}.`
      );
    }
    return { ...preset, slug: format };
  }

  if (
    format &&
    typeof format === "object" &&
    Number.isInteger(format.width) &&
    Number.isInteger(format.height)
  ) {
    const { width, height } = format;
    if (
      width < MIN_DIM || width > MAX_DIM ||
      height < MIN_DIM || height > MAX_DIM
    ) {
      throw new Error(
        `Dimensions hors limites [${MIN_DIM}, ${MAX_DIM}] : ${width}x${height}`
      );
    }
    // x264 exige des dimensions paires
    if (width % 2 !== 0 || height % 2 !== 0) {
      throw new Error(`Dimensions doivent être paires : ${width}x${height}`);
    }
    return {
      label: "custom",
      width,
      height,
      default_fontsize: Math.round(Math.min(width, height) * 0.045),
      default_margin_v: Math.round(height * 0.07),
      alignment: 2,
      slug: "custom",
    };
  }

  throw new Error(
    `Format invalide. Attendu: slug parmi ${Object.keys(PRESETS).join(", ")} ou {width,height}.`
  );
}

function listFormats() {
  return Object.entries(PRESETS).map(([slug, p]) => ({
    slug,
    label: p.label,
    width: p.width,
    height: p.height,
  }));
}

module.exports = { resolveFormat, listFormats, PRESETS };
