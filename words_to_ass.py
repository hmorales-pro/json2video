"""
Convertit un JSON de timestamps Whisper (verbose_json, word-level) en ASS
avec animation karaoké mot-à-mot (balises \\k).

Usage :
    python words_to_ass.py <whisper.json> <out.ass> [--style style.json]

Le JSON `style.json` (optionnel) peut contenir :
    {
      "fontname": "DejaVu Sans",
      "fontsize": 64,
      "primary_color":   [r,g,b,a],
      "secondary_color": [r,g,b,a],
      "outline_color":   [r,g,b,a],
      "back_color":      [r,g,b,a],
      "alignment": 2,                  // ASS 1..9
      "margin_v": 240,
      "outline": 3,
      "shadow": 2,
      "bold": true,
      "max_words_per_line": 4
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pysubs2


DEFAULT_STYLE = {
    "fontname": "DejaVu Sans",
    "fontsize": 56,
    "primary_color": [255, 255, 255, 0],
    "secondary_color": [0, 230, 255, 0],
    "outline_color": [0, 0, 0, 0],
    "back_color": [0, 0, 0, 160],
    "alignment": 2,
    "margin_v": 80,
    "outline": 3,
    "shadow": 2,
    "bold": True,
    "max_words_per_line": 6,
}

PAUSE_BREAK_SEC = 0.6


def _color(rgba) -> pysubs2.Color:
    r, g, b, a = rgba
    return pysubs2.Color(r, g, b, a)


def _apply_style(subs: pysubs2.SSAFile, s: dict) -> None:
    # PlayResX/Y : OBLIGATOIRE — sinon libass scale tout par le ratio
    # vidéo / 384x288 (default), donnant des sous-titres x6 trop grands.
    if s.get("play_res_x"):
        subs.info["PlayResX"] = str(int(s["play_res_x"]))
    if s.get("play_res_y"):
        subs.info["PlayResY"] = str(int(s["play_res_y"]))
    # ScaledBorderAndShadow=no garantit que outline/shadow restent en pixels
    # à l'échelle native (pas mis à l'échelle par PlayRes).
    subs.info["ScaledBorderAndShadow"] = "yes"

    style = pysubs2.SSAStyle()
    style.fontname = s["fontname"]
    style.fontsize = s["fontsize"]
    style.primarycolor = _color(s["primary_color"])
    style.secondarycolor = _color(s["secondary_color"])
    style.outlinecolor = _color(s["outline_color"])
    style.backcolor = _color(s["back_color"])
    style.bold = bool(s["bold"])
    style.outline = s["outline"]
    style.shadow = s["shadow"]
    style.alignment = pysubs2.Alignment(s["alignment"])
    style.marginv = s["margin_v"]
    subs.styles["Default"] = style


def _group_words(
    words: list[dict],
    max_words: int,
    pause_break: float = PAUSE_BREAK_SEC,
) -> Iterable[list[dict]]:
    line: list[dict] = []
    for w in words:
        if line and (w["start"] - line[-1]["end"]) >= pause_break:
            yield line
            line = []
        line.append(w)
        if len(line) >= max_words:
            yield line
            line = []
    if line:
        yield line


def _ass_text_for_line(line: list[dict]) -> str:
    parts: list[str] = []
    for i, w in enumerate(line):
        duration_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
        word = (w.get("word") or "").strip()
        if not word:
            continue
        sep = "" if i == 0 else " "
        parts.append(f"{sep}{{\\k{duration_cs}}}{word}")
    return "".join(parts)


def _extract_words(data: dict) -> list[dict]:
    words = data.get("words") or []
    if not words and "segments" in data:
        for seg in data["segments"]:
            words.extend(seg.get("words", []))
    out = []
    for w in words:
        if "start" not in w or "end" not in w:
            continue
        out.append(
            {
                "word": w.get("word") or w.get("text") or "",
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
        )
    return out


def whisper_words_to_ass(
    words_json_path: str, ass_path: str, style_overrides: dict | None = None
) -> None:
    data = json.loads(Path(words_json_path).read_text(encoding="utf-8"))
    words = _extract_words(data)
    if not words:
        raise ValueError("Aucun mot avec timestamps dans le JSON Whisper.")

    s = {**DEFAULT_STYLE, **(style_overrides or {})}

    subs = pysubs2.SSAFile()
    _apply_style(subs, s)

    for line in _group_words(words, max_words=int(s["max_words_per_line"])):
        start_ms = int(line[0]["start"] * 1000)
        end_ms = int(line[-1]["end"] * 1000)
        if end_ms <= start_ms:
            end_ms = start_ms + 200
        event = pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=_ass_text_for_line(line),
            style="Default",
        )
        subs.events.append(event)

    subs.save(ass_path, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper word-level JSON → ASS karaoké")
    p.add_argument("words_json")
    p.add_argument("out_ass")
    p.add_argument("--style", help="Chemin vers un JSON de style (optionnel).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    style_override = None
    if args.style:
        style_override = json.loads(Path(args.style).read_text(encoding="utf-8"))
    whisper_words_to_ass(args.words_json, args.out_ass, style_override)
    print(f"OK : {args.out_ass}")
