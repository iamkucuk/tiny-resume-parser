"""Draw the README figures into docs/assets/.

    python docs/assets/make_figures.py

Numbers come from the research record (docs/research/journal.md, 2026-10-15, 2026-10-16 and 2026-11-06) and are written out below so a
change is one edit here. The example figure uses docs/assets/example_page.jpg and ResuVL-L's answer
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
    ("ResuVL-L", "0.87B", 0.845, 0.916),
    ("ResuVL-L v0.1", "0.94B", 0.735, 0.840),
    ("ResuVL-M", "0.51B", 0.681, 0.791),
    ("ResuVL-S", "0.26B", 0.652, 0.777),
]
TEACHER = {"lenient": 0.912, "content": 0.967}  # a second GLM-5.3-Flash labelling run vs the 50 reviewed dev gold pages
# Where ResuVL-L v0.1 loses gold values (share of all gold values; journal 2026-10-15 ~12:50).
LOST = [
    ("left out", 10.6),
    ("page not parsed", 6.7),
    ("joined with other text", 4.6),
    ("written in part", 3.1),
    ("misread characters", 2.9),
    ("under another field", 1.8),
]

# ResuVL-L decoding time on one H100, bf16, first 30 validation pages; each pair (plain, drafted) timed in one job
# (journal 2026-11-06 03:00, MTPdraftK: totals 530.1 / 277.8, 405.0 / 177.7, 406.3 / 165.0 s).
SPEED = [  # (drafted tokens per step, plain s per page, drafted s per page)
    (1, 530.1 / 30, 277.8 / 30),
    (2, 405.0 / 30, 177.7 / 30),
    (3, 406.3 / 30, 165.0 / 30),
]

# Lenient F1 on the 50 gold / 79 validation development pages after each step (docs/findings.md; journal 2026-10-13 to 2026-10-16).
STEPS = [  # (what was measured, gold, validation); LoRA's validation score is not stated in the findings
    ("LoRA fine-tune, best run", 0.65, None),
    ("Full fine-tune (3-seed mean)", 0.684, 0.671),
    ("+ image noise, vision encoder trained (3-seed mean)", 0.706, 0.708),
    ("Average of 10 runs = ResuVL-L v0.1", 0.732, 0.724),
    ("Newer base model (3-seed mean)", 0.835, 0.806),
    ("Average of 3 runs = ResuVL-L", 0.858, 0.822),
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
        "Gold values ResuVL-L v0.1 loses (share of all gold values)",
        loc="left",
        fontsize=11,
        color=t["text"],
    )
    style(ax, t)
    ax.tick_params(axis="y", labelcolor=t["text"])
    fig.tight_layout()
    fig.savefig(ASSETS / "lost-values.svg", facecolor="white")
    plt.close(fig)


def speed_chart(theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    y = 0
    ticks, labels = [], []
    for k, plain, drafted in SPEED[::-1]:
        ax.barh(y + 0.2, plain, height=0.38, color=t["grid"])
        ax.barh(y - 0.2, drafted, height=0.38, color=t["bar"])
        ax.text(plain + 0.2, y + 0.2, f"{plain:.1f} s  plain", va="center", fontsize=9.5, color=t["muted"])
        ax.text(
            drafted + 0.2,
            y - 0.2,
            f"{drafted:.1f} s  drafted, {plain / drafted:.2f}x faster",
            va="center",
            fontsize=9.5,
            color=t["text"],
        )
        ticks.append(y)
        labels.append(f"{k} drafted token{'s' if k > 1 else ''}\nper step")
        y += 1.2
    ax.set_yticks(ticks, labels)
    ax.set_xlim(0, 27)
    ax.set_title(
        "ResuVL-L seconds per CV page on an H100 GPU (lower is faster)", loc="left", fontsize=11, color=t["text"]
    )
    style(ax, t)
    ax.tick_params(axis="y", labelcolor=t["text"])
    fig.text(
        0.01,
        0.01,
        "30 validation pages, bf16, one page at a time. Each pair was timed in one job; same field F1 with and without drafts.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ASSETS / "speed.svg", facecolor="white")
    plt.close(fig)


def steps_chart(theme: str) -> None:
    """Dev scores after each step of the recipe (docs/findings.md, "How the recipe developed" and "A newer base model")."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ys = list(range(len(STEPS)))[::-1]
    for y, (_label, gold, val) in zip(ys, STEPS):
        if val is not None:
            ax.plot([gold, val], [y, y], color=t["grid"], linewidth=3, zorder=1)
            ax.scatter([val], [y], s=70, color=t["ceiling"], zorder=2)
        ax.scatter([gold], [y], s=70, color=t["bar"], zorder=3)
        dots = [(gold, f"{gold:.3f}" if val is not None else "about 0.65", t["bar"])]
        if val is not None:
            dots.append((val, f"{val:.3f}", t["ceiling"]))
        dots.sort(key=lambda d: d[0])  # the lower score is labelled on the left, the higher on the right
        for i, (x, text, color) in enumerate(dots):
            left = len(dots) == 2 and i == 0
            ax.text(
                x + (-0.005 if left else 0.005),
                y,
                text,
                va="center",
                ha="right" if left else "left",
                fontsize=9.5,
                color=color,
            )
    ax.axhline(1.5, color=t["grid"], linewidth=1, linestyle=(0, (2, 3)))
    ax.set_yticks(ys, [s[0] for s in STEPS])
    ax.set_xlim(0.62, 0.9)
    ax.set_title("Lenient field F1 after each step of the recipe", loc="left", fontsize=11, color=t["text"])
    style(ax, t)
    ax.tick_params(axis="y", labelcolor=t["text"])
    ax.scatter([], [], s=70, color=t["bar"], label="50 gold pages")
    ax.scatter([], [], s=70, color=t["ceiling"], label="79 validation pages")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=t["muted"])
    fig.text(
        0.01,
        0.01,
        "Development pages only. Above the dotted line: ResuVL-L v0.1's base model. Below it: the newer base model, same recipe.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ASSETS / "steps.svg", facecolor="white")
    plt.close(fig)


def drafting_diagram(theme: str) -> None:
    """How drafted decoding keeps the greedy answer: schematic with placeholder tokens, not a real answer."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, text, fc="white", ec=None, tc=None, w=1.0):
        ax.add_patch(plt.Rectangle((x, y), w, 0.55, facecolor=fc, edgecolor=ec or t["grid"], linewidth=1.5, zorder=2))
        ax.text(x + w / 2, y + 0.275, text, ha="center", va="center", fontsize=10.5, color=tc or t["text"], zorder=3)

    rows = [
        (2.75, "1  The head drafts", "next 3 tokens"),
        (1.7, "2  One pass checks", "the model's own choice"),
        (0.65, "3  Keep", "the matching drafts"),
    ]
    for y, head, sub in rows:
        ax.text(0.0, y + 0.38, head, fontsize=11, fontweight="bold", color=t["text"], va="center")
        ax.text(0.0, y + 0.1, sub, fontsize=9.5, color=t["muted"], va="center")
    x0, gap = 2.6, 1.2
    drafts = ["draft 1", "draft 2", "draft 3"]
    choices = ["choice 1", "choice 2", "choice 3"]
    for i in range(3):
        box(x0 + i * gap, 2.75, drafts[i], fc="#eef3fb", ec=t["bar"])
        box(x0 + i * gap, 1.7, choices[i], fc="white")
    marks = [("✓ same", "#1a7f4b"), ("✓ same", "#1a7f4b"), ("✗ differs", "#c0392b")]
    for i, (m, c) in enumerate(marks):
        ax.text(x0 + i * gap + 0.5, 1.5, m, ha="center", va="top", fontsize=9.5, color=c)
        ax.annotate(
            "",
            xy=(x0 + i * gap + 0.5, 2.27),
            xytext=(x0 + i * gap + 0.5, 2.73),
            arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.2),
            zorder=1,
        )
    box(x0, 0.65, "draft 1", fc="#eef3fb", ec=t["bar"])
    box(x0 + gap, 0.65, "draft 2", fc="#eef3fb", ec=t["bar"])
    box(x0 + 2 * gap, 0.65, "choice 3", fc="white", ec=t["ceiling"], tc=t["text"])
    ax.text(
        x0 + 3 * gap + 0.1,
        0.925,
        "Choice 3 replaces draft 3.\n3 tokens from one pass, the\nsame as plain decoding gives",
        fontsize=9.5,
        color=t["text"],
        va="center",
    )
    ax.text(
        0.0,
        0.0,
        "Schematic: placeholder tokens. Measured on 30 validation pages with 3 drafts per step: 3.19 tokens per model pass, against 1 without drafts.",
        fontsize=8.5,
        color=t["muted"],
    )
    fig.savefig(ASSETS / "drafting.svg", facecolor="white", bbox_inches="tight")
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
        speed_chart(theme)
        steps_chart(theme)
        drafting_diagram(theme)
    example_figure()
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")) + ["example.png"])
