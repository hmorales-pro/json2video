/**
 * Charge un template par son nom, fusionne avec les overrides de la requête,
 * et expose une config normalisée prête pour le moteur de rendu.
 */
const fs = require("fs");
const path = require("path");

const TEMPLATES_DIR = path.resolve(__dirname, "..", "templates");

const TEMPLATE_CACHE = new Map();

function _load(name) {
  if (TEMPLATE_CACHE.has(name)) return TEMPLATE_CACHE.get(name);
  const filePath = path.join(TEMPLATES_DIR, `${name}.json`);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Template inconnu: "${name}".`);
  }
  const raw = JSON.parse(fs.readFileSync(filePath, "utf8"));
  TEMPLATE_CACHE.set(name, raw);
  return raw;
}

function listTemplates() {
  return fs
    .readdirSync(TEMPLATES_DIR)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      const data = _load(path.basename(f, ".json"));
      return {
        name: data.name,
        label: data.label,
        format: data.format,
        subtitle_mode: data.subtitle?.mode,
      };
    });
}

/** Deep merge : src override target, sans muter. */
function deepMerge(target, src) {
  if (Array.isArray(src)) return src.slice();
  if (src && typeof src === "object") {
    const out = { ...(target || {}) };
    for (const [k, v] of Object.entries(src)) {
      out[k] = deepMerge(target?.[k], v);
    }
    return out;
  }
  return src === undefined ? target : src;
}

/**
 * Résout la config finale :
 *   1) part du template (si fourni)
 *   2) écrase avec les champs envoyés par le client
 *   3) renvoie une config "plate" prête à valider
 */
function resolveConfig(userConfig = {}) {
  const templateName = userConfig.template;
  const base = templateName ? _load(templateName) : {};
  // On ne propage pas le nom du template dans la config finale.
  const { template: _drop, ...userOverrides } = userConfig;
  return deepMerge(base, userOverrides);
}

module.exports = { resolveConfig, listTemplates };
