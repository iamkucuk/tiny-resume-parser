"""Draw the README figures into docs/assets/.

    python docs/assets/make_figures.py

Numbers come from the research record (docs/research/journal.md, 2026-10-15 and 2026-10-16) and are written out below so a
change is one edit here. The example figure uses docs/assets/example_page.jpg and ResuVL-L's (Qwen3.5-0.8B) answer
for that page (docs/assets/example_answer.json).
"""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ASSETS = Path(__file__).resolve().parent

# Lenient and content field F1 on the 204 held-out test pages, raw answers, transformers 5.5.4 (slurm/test.slurm, 2026-10-16).
RESULTS = [  # (student, parameters, lenient F1, content F1)
    ("Qwen3.5-0.8B", "0.87B", 0.845, 0.916),
    ("InternVL3-1B", "0.94B", 0.735, 0.840),
    ("SmolVLM-500M", "0.51B", 0.681, 0.791),
    ("granite-docling-258M", "0.26B", 0.652, 0.777),
]
TEACHER = {"lenient": 0.912, "content": 0.967}  # a second GLM-5.3-Flash labelling run vs the 50 reviewed dev gold pages
# Where the best InternVL model loses gold values (share of all gold values; journal 2026-10-15 ~12:50).
LOST = [
    ("left out", 10.6),
    ("page not parsed", 6.7),
    ("joined with other text", 4.6),
    ("written in part", 3.1),
    ("misread characters", 2.9),
    ("under another field", 1.8),
]

# One light version on a white card: GitHub picks <picture> variants by the operating-system theme, which can differ
# from the GitHub theme and leave light text on a light page. A white card reads on both page themes.
THEMES = {"light": dict(bar="#2a78d6", text="#0b0b0b", muted="#52514e", grid="#d9d8d4", ceiling="#eb6834")}


def style(ax, t):
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def results_chart(theme: str) -> None:
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    names = [f"{n}  ({p})" for n, p, _, _ in RESULTS][::-1]
    for ax, (metric, col, title) in zip(axes, [("lenient", 2, "Lenient field F1"), ("content", 3, "Content F1")]):
        values = [r[col] for r in RESULTS][::-1]
        ax.barh(names, values, height=0.5, color=t["bar"])
        for y, v in enumerate(values):  # inside the bar end, so no label meets the teacher line
            ax.text(v - 0.015, y, f"{v:.3f}", va="center", ha="right", fontsize=10, color="white")
        ax.axvline(TEACHER[metric], color=t["ceiling"], linewidth=2, linestyle=(0, (4, 3)))
        ax.set_xlim(0, 1)
        ax.set_title(f"{title}   (dashed line: {TEACHER[metric]:.3f})", loc="left", fontsize=11, color=t["text"])
        style(ax, t)
    axes[0].tick_params(axis="y", labelcolor=t["text"])
    fig.text(
        0.01,
        0.01,
        "204 held-out test pages. Dashed line: the labelling model, run a second time on 50 reviewed "
        "development pages, scored against them.",
        fontsize=9,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / "results.svg", facecolor="white")
    plt.close(fig)


def lost_chart(theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    labels, values = [n for n, _ in LOST][::-1], [v for _, v in LOST][::-1]
    ax.barh(labels, values, height=0.55, color=t["bar"])
    for y, v in enumerate(values):
        ax.text(v + 0.2, y, f"{v:.1f} %", va="center", fontsize=10, color=t["text"])
    ax.set_xlim(0, 13)
    ax.set_title(
        "Gold values the earlier InternVL average loses (share of all gold values)",
        loc="left",
        fontsize=11,
        color=t["text"],
    )
    style(ax, t)
    ax.tick_params(axis="y", labelcolor=t["text"])
    fig.tight_layout()
    fig.savefig(ASSETS / "lost-values.svg", facecolor="white")
    plt.close(fig)


def compact_json(x, indent: int = 0) -> str:
    """JSON with one key per line but lists of plain values on one line, so more of the answer fits."""
    pad = "  " * indent
    if isinstance(x, dict):
        items = [f'{pad}  "{k}": {compact_json(v, indent + 1).lstrip()}' for k, v in x.items()]
        return pad + "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(x, list) and any(isinstance(v, (dict, list)) for v in x):
        return pad + "[\n" + ",\n".join(compact_json(v, indent + 1) for v in x) + "\n" + pad + "]"
    return pad + json.dumps(x, ensure_ascii=False)


def example_figure() -> None:
    """The page image next to an excerpt of the model's JSON answer for it."""
    page = Image.open(ASSETS / "example_page.jpg")
    answer = json.loads((ASSETS / "example_answer.json").read_text())["answer"]
    shown = {
        "basics": {k: answer["basics"][k] for k in ("name", "emails", "phones", "location")},
        "work": [{k: answer["work"][0][k] for k in ("name", "position", "type", "startDate", "endDate")}],
        "education": [
            {k: answer["education"][0].get(k) for k in ("institution", "area", "studyType", "startDate", "endDate")}
        ],
        "skills": answer["skills"][:2],
        "languages": answer["languages"][:3],
    }
    lines = []
    for line in compact_json(shown).splitlines():
        lines += textwrap.wrap(line, 58, subsequent_indent="    ") or [""]
    fig = plt.figure(figsize=(11, 7.2), facecolor="white")
    ax_img = fig.add_axes((0.01, 0.02, 0.42, 0.9))
    ax_img.imshow(page)
    ax_img.axis("off")
    ax_img.set_title("CV page (image)", fontsize=12, loc="left")
    ax_txt = fig.add_axes((0.47, 0.02, 0.52, 0.9))
    ax_txt.axis("off")
    ax_txt.set_title("JSON written by ResuVL-L, 0.87B (excerpt)", fontsize=12, loc="left")
    ax_txt.text(0, 1, "\n".join(lines[:50] + ["  ..."]), va="top", family="monospace", fontsize=8.2, color="#1f2328")
    fig.savefig(ASSETS / "example.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    for theme in THEMES:
        results_chart(theme)
        lost_chart(theme)
    example_figure()
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")) + ["example.png"])
