/**
 * Dry-run du moteur de rendu : valide config + construit filter_complex
 * sans appeler FFmpeg.
 */
const { resolveConfig } = require("../lib/templates");
const { validateConfig } = require("../lib/validate");
const { buildFilterComplex } = require("../lib/render");

function header(label) {
  console.log("\n===", label, "===");
}

header("TEST 1 : Reels karaoké, 3 images");
let cfg = resolveConfig({
  template: "reels-karaoke",
  media: [
    { duration: 2.0, fit: "cover" },
    { duration: 3.5, fit: "contain" },
    { duration: 1.5, fit: "stretch" },
  ],
});
let v = validateConfig(cfg);
let items = v.media.map((m) => ({ ...m, __sourceType: "image", __duration_sec: m.duration }));
console.log("Format       :", v.format.width + "x" + v.format.height);
console.log("filter_complex:");
console.log(buildFilterComplex(items, v.format, v.fps, "/tmp/sub.ass"));

header("TEST 2 : YouTube captions, 1 image + 1 vidéo trimée");
cfg = resolveConfig({
  template: "youtube-captions",
  media: [
    { duration: 4 },
    { trim_start: 5, trim_end: 12 },
  ],
});
v = validateConfig(cfg);
items = [
  { ...v.media[0], __sourceType: "image", __duration_sec: 4 },
  { ...v.media[1], __sourceType: "video", __duration_sec: 7 },
];
console.log("filter_complex:");
console.log(buildFilterComplex(items, v.format, v.fps, "/tmp/sub.ass"));

header("TEST 3 : Reels avec transitions xfade fade 0.3s + overlay texte");
cfg = resolveConfig({
  template: "reels-karaoke",
  media: [
    { duration: 3 },
    { duration: 3 },
    { duration: 3 },
  ],
  transitions: { type: "fade", duration: 0.3 },
  overlays: [
    { type: "text", text: "Sponsorisé", start: 0, end: 2, x: "right", y: "top", fontsize: 32, color: "#FFFFFF", bg_color: "#FF0000", bg_opacity: 0.8 },
  ],
});
v = validateConfig(cfg);
items = v.media.map((m) => ({ ...m, __sourceType: "image", __duration_sec: m.duration }));
console.log("filter_complex:");
console.log(buildFilterComplex(items, v.format, v.fps, "/tmp/sub.ass", {
  transitions: v.transitions,
  overlays: v.overlays,
  overlayTextFiles: ["/tmp/o0.txt"],
}));

header("TEST 4 : YouTube + mix musique + ducking");
cfg = resolveConfig({
  template: "youtube-captions",
  media: [{ duration: 5 }],
  audio: { music_volume: 0.25, voice_volume: 1.1, ducking: true },
});
v = validateConfig(cfg);
items = [{ ...v.media[0], __sourceType: "image", __duration_sec: 5 }];
console.log("filter_complex:");
console.log(buildFilterComplex(items, v.format, v.fps, "/tmp/sub.ass", {
  transitions: v.transitions,
  audioInputs: [
    { label: "1:a:0", role: "voice" },
    { label: "2:a:0", role: "music" },
  ],
  audioCfg: v.audio,
}));

header("TEST 5 : Custom format 1280x720, slide transition, 2 médias, pas de subs");
cfg = {
  format: { width: 1280, height: 720 },
  fps: 24,
  subtitle: { mode: "none" },
  transitions: { type: "slideleft", duration: 0.5 },
  media: [{ duration: 2 }, { duration: 3 }],
};
v = validateConfig(cfg);
items = v.media.map((m) => ({ ...m, __sourceType: "image", __duration_sec: m.duration }));
console.log("filter_complex:");
console.log(buildFilterComplex(items, v.format, v.fps, null, {
  transitions: v.transitions,
}));

header("TEST 6 : Validation rejette les configs malveillantes");
function expectReject(label, fn) {
  try { fn(); console.log("  ✗ ACCEPTÉ :", label); }
  catch (e) { console.log("  ✓ rejeté :", label, "→", e.message.slice(0, 75)); }
}
expectReject("fontname avec injection", () =>
  validateConfig({ format: "9:16", subtitle: { mode: "karaoke", style: { fontname: "A; rm -rf" } }, media: [{}] })
);
expectReject("transition type exotique", () =>
  validateConfig({ format: "9:16", transitions: { type: "evil-filter" }, media: [{}, {}] })
);
expectReject("overlay texte avec NUL byte", () =>
  validateConfig({ format: "9:16", overlays: [{ type: "text", text: "ok\x00evil" }], media: [{}] })
);
expectReject("music_volume hors plage", () =>
  validateConfig({ format: "9:16", audio: { music_volume: 99 }, media: [{}] })
);
expectReject("position overlay invalide", () =>
  validateConfig({ format: "9:16", overlays: [{ type: "text", text: "x", x: "diagonal" }], media: [{}] })
);
expectReject("transition duration trop longue", () =>
  validateConfig({ format: "9:16", transitions: { type: "fade", duration: 5 }, media: [{}, {}] })
);
