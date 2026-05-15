/**
 * Wrappers Whisper (OpenAI) :
 *   - transcribeSrt(file, lang)        → SRT brut
 *   - transcribeWordLevel(file, lang)  → JSON verbose avec timestamps mot-à-mot
 */
const axios = require("axios");
const FormData = require("form-data");
const fsSync = require("fs");

// URL paramétrable pour pouvoir basculer vers un serveur Whisper self-hosted
// (whisper.cpp HTTP server, faster-whisper-server, LocalAI, etc.) sans
// modifier le code. Le serveur cible doit exposer une API compatible OpenAI
// /v1/audio/transcriptions.
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
const OPENAI_URL = `${OPENAI_BASE_URL}/audio/transcriptions`;

function _baseFormData(file, language) {
  const fd = new FormData();
  fd.append("file", fsSync.createReadStream(file.path), file.originalname || "audio.mp3");
  fd.append("model", "whisper-1");
  if (language) fd.append("language", language.split("-")[0]); // "fr-FR" → "fr"
  return fd;
}

async function _postWhisper(formData, apiKey) {
  const contentLength = await new Promise((resolve, reject) => {
    formData.getLength((err, length) => (err ? reject(err) : resolve(length)));
  });
  try {
    const response = await axios.post(OPENAI_URL, formData, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        ...formData.getHeaders(),
        "Content-Length": contentLength,
      },
      maxBodyLength: Infinity,
      timeout: 180_000,
    });
    return response.data;
  } catch (err) {
    // Transforme les erreurs OpenAI en messages explicites côté serveur.
    const status = err.response?.status;
    const openaiMsg =
      err.response?.data?.error?.message ||
      err.response?.data?.error ||
      err.message;
    const hint =
      status === 401
        ? " — OPENAI_API_KEY invalide ou révoquée. Vérifie ta clé sur https://platform.openai.com/api-keys"
        : status === 429
        ? " — Quota OpenAI dépassé ou rate-limit."
        : status === 400
        ? " — Requête refusée par OpenAI (format audio non supporté ?)"
        : "";
    const e = new Error(`OpenAI Whisper ${status || "?"}: ${openaiMsg}${hint}`);
    e.status = status === 401 ? 502 : 500; // 502 = problème de dépendance externe
    e.openaiStatus = status;
    throw e;
  }
}

function transcribeSrt(apiKey) {
  return async (file, language) => {
    const fd = _baseFormData(file, language);
    fd.append("response_format", "srt");
    return _postWhisper(fd, apiKey);
  };
}

function transcribeWordLevel(apiKey) {
  return async (file, language) => {
    const fd = _baseFormData(file, language);
    fd.append("response_format", "verbose_json");
    fd.append("timestamp_granularities[]", "word");
    return _postWhisper(fd, apiKey);
  };
}

module.exports = { transcribeSrt, transcribeWordLevel };
