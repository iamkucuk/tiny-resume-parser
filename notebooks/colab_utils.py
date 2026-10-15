"""Inline HTML views for tracking training in Colab / Jupyter.

    from colab_utils import summary, TrainingDashboard, generations_report
    summary("Data", {"images": 1726}, ok=True)                 # one status card
    tiny_resume_parser.train.main(args, callbacks=[TrainingDashboard()])     # live dashboard instead of tqdm
    generations_report(out / "generations.jsonl")              # valid/invalid table with answers

Plain HTML + inline SVG, no extra packages. Colours are semi-transparent so the cards read on
Colab's light and dark themes.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

from IPython.display import HTML, display
from transformers import TrainerCallback

_CSS = """<style>
.cvp{font-family:system-ui,sans-serif;font-size:13px;border:1px solid rgba(127,127,127,.35);
 border-radius:8px;padding:10px 14px;margin:6px 0;max-width:860px}
.cvp h4{margin:0 0 8px;font-size:14px}
.cvp .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px 14px}
.cvp .k{opacity:.65;font-size:11px}.cvp .v{font-weight:600;font-variant-numeric:tabular-nums}
.cvp .bar{height:8px;border-radius:4px;background:rgba(127,127,127,.25);overflow:hidden;margin:6px 0 10px}
.cvp .bar>div{height:100%;background:#3b82f6}
.cvp .ok{color:#16a34a}.cvp .bad{color:#dc2626}.cvp .warn{color:#d97706}
.cvp table{border-collapse:collapse;width:100%}.cvp td,.cvp th{padding:3px 8px;text-align:left;
 border-bottom:1px solid rgba(127,127,127,.2);vertical-align:top}
.cvp pre{white-space:pre-wrap;word-break:break-all;max-height:240px;overflow:auto;font-size:11px;margin:4px 0}
</style>"""


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


def _grid(items: dict) -> str:
    return (
        '<div class="grid">'
        + "".join(
            f'<div><div class="k">{html.escape(k)}</div><div class="v">{_fmt(v)}</div></div>' for k, v in items.items()
        )
        + "</div>"
    )


def _status(ok: bool | None) -> str:
    return {True: '<span class="ok">✔</span> ', False: '<span class="bad">✘</span> ', None: ""}[ok]


def summary(title: str, items: dict, ok: bool | None = None) -> None:
    """One card: title with ✔/✘ and a grid of key → value."""
    display(HTML(f'{_CSS}<div class="cvp"><h4>{_status(ok)}{html.escape(title)}</h4>{_grid(items)}</div>'))


def _sparkline(ys: list[float], w: int = 520, h: int = 90) -> str:
    if len(ys) < 2:
        return '<div class="k">loss chart appears after 2 logged steps</div>'
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    pts = " ".join(
        f"{i * (w - 10) / (len(ys) - 1) + 5:.1f},{h - 5 - (y - lo) * (h - 10) / span:.1f}" for i, y in enumerate(ys)
    )
    return (
        f'<svg width="{w}" height="{h}" style="display:block"><polyline fill="none" stroke="#3b82f6" '
        f'stroke-width="1.8" points="{pts}"/><text x="5" y="12" font-size="10" fill="currentColor" opacity=".6">'
        f'max {hi:.3g}</text><text x="5" y="{h - 2}" font-size="10" fill="currentColor" opacity=".6">min {lo:.3g}</text></svg>'
    )


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TrainingDashboard(TrainerCallback):
    """Live card for a transformers Trainer run: progress bar, loss chart, lr, grad norm, ETA, GPU memory."""

    def __init__(self, title: str = "Training"):
        self.title = title
        self.handle = None
        self.t0 = 0.0
        self.logs: list[dict] = []
        self.done = False

    def _render(self, state) -> None:
        import torch

        step, total = state.global_step, max(state.max_steps, 1)
        elapsed = time.time() - self.t0
        per_step = elapsed / step if step else None
        losses = [x["loss"] for x in self.logs if "loss" in x]
        last = self.logs[-1] if self.logs else {}
        nan = any(l != l for l in losses)  # NaN check
        items = {
            "step": f"{step} / {total}",
            "epoch": f"{state.epoch or 0:.2f}",
            "loss (last)": losses[-1] if losses else "–",
            "loss (first)": losses[0] if losses else "–",
            "learning rate": last.get("learning_rate", "–"),
            "grad norm": last.get("grad_norm", "–"),
            "elapsed": _hms(elapsed),
            "time / step": f"{per_step:.1f} s" if per_step else "–",
            "time left": _hms(per_step * (total - step)) if per_step else "–",
            "memory": f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB peak GPU"
            if torch.cuda.is_available()
            else "CPU run",
        }
        ok = False if nan else (True if self.done else None)
        state_txt = "finished" if self.done else "running"
        body = (
            f"<h4>{_status(ok)}{html.escape(self.title)} — {state_txt}{' — loss is NaN' if nan else ''}</h4>"
            f'<div class="bar"><div style="width:{100 * step / total:.1f}%"></div></div>{_grid(items)}'
            f'<div style="margin-top:8px">{_sparkline(losses)}</div>'
        )
        content = HTML(f'{_CSS}<div class="cvp">{body}</div>')
        if self.handle is None:
            self.handle = display(content, display_id=True)
        else:
            self.handle.update(content)

    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.logs, self.done = time.time(), [], False
        self._render(state)

    def on_step_end(self, args, state, control, **kw):
        self._render(state)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.logs.append(dict(logs))
        self._render(state)

    def on_train_end(self, args, state, control, **kw):
        self.done = True
        self._render(state)


def generations_report(path: Path, preview_chars: int = 1500) -> None:
    """Table of generated answers: ✔/✘ against the schema, length, error, and the answer text on click."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    valid = sum(r["result"] == "valid" for r in rows)
    trs = []
    for r in rows:
        ok = r["result"] == "valid"
        err = "" if ok else html.escape(r["result"][:160])
        trs.append(
            f"<tr><td>{_status(ok)}</td><td>{html.escape(r['cv_id'])} p{r['page']}</td><td>{len(r['answer']):,}</td>"
            f'<td>{err}<details><summary class="k">answer</summary><pre>{html.escape(r["answer"][:preview_chars])}'
            f"</pre></details></td></tr>"
        )
    head = f"<h4>{_status(valid == len(rows))}Generated answers: {valid} / {len(rows)} valid, {len({r['answer'] for r in rows})} distinct</h4>"
    display(
        HTML(
            f'{_CSS}<div class="cvp">{head}<table><tr><th></th><th>page</th><th>chars</th><th>error / answer</th></tr>'
            f"{''.join(trs)}</table></div>"
        )
    )
