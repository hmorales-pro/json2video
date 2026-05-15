"""
Convertit un SRT en ASS avec un style paramétrable (captions mode).

Usage :
    python convert_srt_to_ass.py <in.srt> <out.ass> [--style style.json]

Le JSON `style.json` accepte les mêmes clés que words_to_ass.py
(fontname, fontsize, primary_color, outline_color, back_color, alignment,
margin_v, outline, shadow, bold).

Cette version corrige le bug historique de duplication des mots
(`for word in words:` réécrivait deux fois la chaîne).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pysubs2


DEFAULT_STYLE = {
    "fontname": "DejaVu Sans",
    "fontsize": 48,
    "primary_color": [255, 255, 255, 0],
    "outline_color": [0, 0, 0, 0],
    "back_color": [0, 0, 0, 160],
    "alignment": 2,
    "margin_v": 80,
    "outline": 2,
    "shadow": 1,
    "bold": False,
}


def _color(rgba) -> pysubs2.Color:
    r, g, b, a = rgba
    return pysubs2.Color(r, g, b, a)


def _apply_style(subs: pysubs2.SSAFile, s: dict) -> None:
    # PlayResX/Y : OBLIGATOIRE — sinon libass scale tout par défaut (384x288).
    if s.get("play_res_x"):
        subs.info["PlayResX"] = str(int(s["play_res_x"]))
    if s.get("play_res_y"):
        subs.info["PlayResY"] = str(int(s["play_res_y"]))
    subs.info["ScaledBorderAndShadow"] = "yes"

    style = pysubs2.SSAStyle()
    style.fontname = s["fontname"]
    style.fontsize = s["fontsize"]
    style.primarycolor = _color(s["primary_color"])
    style.outlinecolor = _color(s["outline_color"])
    style.backcolor = _color(s["back_color"])
    style.bold = bool(s["bold"])
    style.outline = s["outline"]
    style.shadow = s["shadow"]
    style.alignment = pysubs2.Alignment(s["alignment"])
    style.marginv = s["margin_v"]
    subs.styles["Default"] = style


def convert_srt_to_ass(
    srt_path: str, ass_path: str, style_overrides: dict | None = None
) -> None:
    subs = pysubs2.load(srt_path, encoding="utf-8")
    s = {**DEFAULT_STYLE, **(style_overrides or {})}
    _apply_style(subs, s)
    for line in subs.events:
        line.style = "Default"
        # On ne remplace pas le texte (le SRT contient déjà des phrases prêtes).
    subs.save(ass_path, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRT → ASS (captions)")
    p.add_argument("in_srt")
    p.add_argument("out_ass")
    p.add_argument("--style", help="Chemin vers un JSON de style (optionnel).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    style_override = None
    if args.style:
        style_override = json.loads(Path(args.style).read_text(encoding="utf-8"))
    convert_srt_to_ass(args.in_srt, args.out_ass, style_override)
    print(f"OK : {args.out_ass}")
