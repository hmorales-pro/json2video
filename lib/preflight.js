/**
 * Vérifications au boot du serveur.
 *
 * - Le binaire ffmpeg est trouvable
 * - Il a été compilé avec le filtre `subtitles` (libass)
 * - Le binaire python3 + le script de génération ASS sont accessibles
 *
 * Si une vérif échoue, on log un message ACTIONNABLE et on retourne false.
 * Le serveur peut toujours démarrer (l'utilisateur peut vouloir tester sans
 * sous-titres) mais saura tout de suite quel chemin de réparation prendre.
 */
const { spawn } = require("child_process");
const fs = require("fs");

const FFMPEG_BIN = process.env.FFMPEG_BIN || "ffmpeg";
const FFPROBE_BIN = process.env.FFPROBE_BIN || "ffprobe";
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";

function _runCapture(bin, args, timeoutMs = 5000) {
  return new Promise((resolve) => {
    const child = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
    child.stdout.on("data", (d) => { out += d.toString(); });
    child.stderr.on("data", (d) => { err += d.toString(); });
    child.on("error", (e) => {
      clearTimeout(timer);
      resolve({ ok: false, code: -1, stdout: "", stderr: e.message });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ ok: code === 0, code, stdout: out, stderr: err });
    });
  });
}

async function checkFfmpeg() {
  const probes = [];

  // 1) ffmpeg lance-t-il quelque chose ?
  const version = await _runCapture(FFMPEG_BIN, ["-version"]);
  if (!version.ok) {
    probes.push({
      level: "fatal",
      msg:
        `FFMPEG_BIN="${FFMPEG_BIN}" introuvable. ` +
        `Installer ffmpeg (brew install ffmpeg) ou pointer FFMPEG_BIN vers le bon chemin.`,
    });
    return probes;
  }

  // 2) Le filtre `subtitles` est-il présent (libass) ?
  const filters = await _runCapture(FFMPEG_BIN, ["-hide_banner", "-filters"]);
  const hasSubtitles = /(^|\s)subtitles\s/m.test(filters.stdout);
  if (!hasSubtitles) {
    // Chercher des binaires alternatifs susceptibles d'avoir libass.
    // On teste UNIQUEMENT ceux qui contiennent effectivement le filtre subtitles
    // (sinon on conseillait à tort le binaire minimal d'Homebrew).
    const allCandidates = [
      "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
      "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
      "/opt/homebrew/bin/ffmpeg",
      "/usr/local/bin/ffmpeg",
      "/usr/bin/ffmpeg",
    ].filter((p) => p !== FFMPEG_BIN && fs.existsSync(p));

    const validCandidates = [];
    for (const c of allCandidates) {
      const r = await _runCapture(c, ["-hide_banner", "-filters"]);
      if (r.ok && /(^|\s)subtitles\s/m.test(r.stdout)) validCandidates.push(c);
    }

    probes.push({
      level: "warn",
      msg:
        `FFMPEG_BIN="${FFMPEG_BIN}" n'inclut pas le filtre 'subtitles' (libass manquant). ` +
        `Les modes subtitle "captions" et "karaoke" échoueront. ` +
        (validCandidates.length
          ? `Binaire(s) avec libass détecté(s) : ${validCandidates.join(", ")}. ` +
            `Ajouter dans .env : FFMPEG_BIN=${validCandidates[0]}`
          : allCandidates.length
          ? `Autres ffmpeg trouvés mais SANS libass : ${allCandidates.join(", ")}. ` +
            `Installer la version complète : brew install ffmpeg-full ` +
            `puis dans .env : FFMPEG_BIN=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`
          : `Installer ffmpeg avec libass : brew install ffmpeg-full ` +
            `puis dans .env : FFMPEG_BIN=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`),
    });
  }

  // 3) ffprobe
  const probeOk = await _runCapture(FFPROBE_BIN, ["-version"]);
  if (!probeOk.ok) {
    probes.push({
      level: "warn",
      msg: `FFPROBE_BIN="${FFPROBE_BIN}" introuvable — la durée des vidéos ne pourra pas être lue.`,
    });
  }

  return probes;
}

async function checkPython() {
  const probes = [];
  const py = await _runCapture(PYTHON_BIN, ["-c", "import pysubs2"]);
  if (!py.ok) {
    probes.push({
      level: "warn",
      msg:
        `PYTHON_BIN="${PYTHON_BIN}" sans le module pysubs2. ` +
        `Installer : pip3 install -r requirements.txt`,
    });
  }
  return probes;
}

async function runPreflight() {
  const all = [
    ...(await checkFfmpeg()),
    ...(await checkPython()),
  ];
  return all;
}

function logPreflight(probes) {
  if (!probes.length) {
    console.log("✓ Preflight OK : ffmpeg+libass, ffprobe, python3+pysubs2.");
    return;
  }
  console.warn("");
  console.warn("══════════ Préflight ══════════");
  for (const p of probes) {
    const tag = p.level === "fatal" ? "✗ FATAL" : "⚠ ATTENTION";
    console.warn(`${tag} : ${p.msg}`);
  }
  console.warn("════════════════════════════════");
  console.warn("");
}

module.exports = { runPreflight, logPreflight };
