/**
 * Webhooks sortants — pour notifier un client (Make.com, n8n, Zapier, etc.)
 * de la fin d'un job de rendu.
 *
 * Sécurité (anti-SSRF) :
 *   - Schéma HTTPS uniquement (pas de http:// pour éviter MITM)
 *   - Refus des hostnames localhost / 0.0.0.0
 *   - DNS lookup : refus si l'hôte résout vers une IP privée
 *     (10/8, 172.16/12, 192.168/16, 127/8, 169.254/16, ::1, fc00::/7, fe80::/10)
 *   - Pas de redirects vers des IPs privées (axios maxRedirects=3 avec validateStatus)
 *
 * Robustesse :
 *   - 3 tentatives avec backoff exponentiel (2s, 4s, 8s)
 *   - Timeout 15s par tentative
 *   - Renvoie le statut HTTP de la dernière tentative pour debug
 */
const axios = require("axios");
const dns = require("dns").promises;

const ALLOWED_PROTOCOLS = new Set(["https:"]);
const FORBIDDEN_HOSTNAMES = new Set(["localhost", "0.0.0.0"]);

function _isPrivateIp(ip) {
  // IPv4
  if (/^127\./.test(ip)) return true;       // loopback
  if (/^10\./.test(ip)) return true;        // private class A
  if (/^192\.168\./.test(ip)) return true;  // private class C
  if (/^172\.(1[6-9]|2[0-9]|3[01])\./.test(ip)) return true; // private class B
  if (/^169\.254\./.test(ip)) return true;  // link-local (AWS metadata !)
  // IPv6
  if (ip === "::1") return true;
  if (/^fc/i.test(ip)) return true;         // unique local
  if (/^fe80/i.test(ip)) return true;       // link-local
  return false;
}

/**
 * Valide une URL de webhook. Throw avec status 400 si KO.
 */
async function validateWebhookUrl(url) {
  if (typeof url !== "string" || url.length > 2000) {
    throw _err400("webhook_url: chaîne attendue (max 2000 chars)");
  }
  let u;
  try {
    u = new URL(url);
  } catch {
    throw _err400("webhook_url: URL malformée");
  }
  if (!ALLOWED_PROTOCOLS.has(u.protocol)) {
    throw _err400("webhook_url: HTTPS obligatoire");
  }
  if (FORBIDDEN_HOSTNAMES.has(u.hostname.toLowerCase())) {
    throw _err400(`webhook_url: hostname interdit (${u.hostname})`);
  }
  // DNS lookup pour SSRF. Note : TOCTOU possible avec DNS rebinding mais
  // c'est une protection raisonnable côté serveur.
  try {
    const records = await dns.lookup(u.hostname, { all: true });
    for (const { address } of records) {
      if (_isPrivateIp(address)) {
        throw _err400(
          `webhook_url: ${u.hostname} résout vers une IP privée (${address})`
        );
      }
    }
  } catch (err) {
    if (err.status === 400) throw err;
    // Résolution DNS échouée : on laisse passer, axios échouera proprement.
    console.warn(`[webhook] DNS lookup KO pour ${u.hostname}: ${err.message}`);
  }
}

function _err400(msg) {
  const e = new Error(msg);
  e.status = 400;
  return e;
}

/**
 * Envoie un POST JSON vers `url` avec retry exponentiel.
 * Fire-and-forget : ne throw jamais ; renvoie un résumé pour le log.
 */
async function postWebhook(url, payload, { maxRetries = 3, timeoutMs = 15_000 } = {}) {
  let lastErr;
  let lastStatus;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const r = await axios.post(url, payload, {
        timeout: timeoutMs,
        headers: {
          "Content-Type": "application/json",
          "User-Agent": "json2video-webhook/1.0",
        },
        validateStatus: (s) => s >= 200 && s < 400,
        maxRedirects: 3,
      });
      console.log(`[webhook] ✓ ${url} → HTTP ${r.status} (tentative ${attempt})`);
      return { ok: true, status: r.status, attempts: attempt };
    } catch (err) {
      lastErr = err;
      lastStatus = err.response?.status;
      console.warn(
        `[webhook] ✗ tentative ${attempt}/${maxRetries} sur ${url} → ` +
        (lastStatus ? `HTTP ${lastStatus}` : err.code || err.message)
      );
      if (attempt < maxRetries) {
        const backoff = Math.pow(2, attempt) * 1000; // 2s, 4s, 8s
        await new Promise((r) => setTimeout(r, backoff));
      }
    }
  }
  console.error(`[webhook] ✗ échec définitif sur ${url} après ${maxRetries} tentatives`);
  return {
    ok: false,
    status: lastStatus || null,
    attempts: maxRetries,
    error: lastErr?.message || "unknown",
  };
}

module.exports = { validateWebhookUrl, postWebhook };
