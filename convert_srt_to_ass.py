import pysubs2
import sys

def convert_srt_to_ass(srt_path, ass_path):
    subs = pysubs2.load(srt_path, encoding="utf-8")
    style = pysubs2.SSAStyle()
    style.fontsize = 40
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)  # Blanc
    style.secondarycolor = pysubs2.Color(255, 0, 0, 0)  # Rouge pour mot prononcé
    style.fontname = "Arial"
    style.bold = True

    for line in subs.events:
        words = line.text.split(" ")
        for i, word in enumerate(words):
            if i == 0:
                line.text = f"{{\\1c&HFF0000&}}{word}{{\\r}}{' '.join(words[1:])}"
            else:
                line.text += f" {word}"

    subs.styles["Default"] = style
    subs.save(ass_path, encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilisation : python convert_srt_to_ass.py <srt_path> <ass_path>")
        sys.exit(1)

    srt_path = sys.argv[1]
    ass_path = sys.argv[2]
    convert_srt_to_ass(srt_path, ass_path)
