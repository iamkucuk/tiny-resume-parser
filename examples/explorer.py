"""Interactive page explorer for the example notebooks: read one CV page with a trained model and see why it scores
what it scores.

    from explorer import Explorer
    Explorer("runs/example/model", train_rows, eval_rows).show()

Tabs: Fields (each value of the answer next to the label value, with its status), Scores (strict, lenient and content
F1), JSON (the repaired answer), Model input (the image tiles and token counts the model receives). Works with
ipywidgets 7 (Colab) and 8. Not part of the package: it only calls tiny_resume_parser functions.
"""

from __future__ import annotations

import base64
import difflib
import html
import io
import json
import math
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, display
from PIL import Image

from tiny_resume_parser.data import page_image
from tiny_resume_parser.generate import generate_pages, load_model, prompt_text
from tiny_resume_parser.metrics import content, f1, lenient, pairs, read_answer
from tiny_resume_parser.models import TRAINED_PATCHES, family, family_name
from tiny_resume_parser.parse import page_images

CSS = """<style>
.tx{font:13px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;color:inherit}
.tx h4{margin:0 0 6px;font-size:14px}
.tx .muted{opacity:.65}
.tx table{border-collapse:collapse;width:100%}
.tx td,.tx th{padding:3px 8px;border-bottom:1px solid rgba(128,128,128,.25);text-align:left;vertical-align:top}
.tx th{font-weight:600;position:sticky;top:0;background:rgba(128,128,128,.18)}
.tx code{font:12px ui-monospace,Menlo,monospace}
.tx .badge{display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;font-weight:600;white-space:nowrap}
.tx .match{background:rgba(26,127,55,.18);color:#1a7f37}
.tx .differs{background:rgba(191,135,0,.2);color:#9a6700}
.tx .extra{background:rgba(207,34,46,.16);color:#cf222e}
.tx .missing{background:rgba(110,119,129,.2);color:#6e7781}
.tx mark{background:rgba(207,34,46,.25);color:inherit;border-radius:2px;padding:0 1px}
.tx .cards{display:flex;gap:10px;flex-wrap:wrap}
.tx .card{flex:1 1 170px;border:1px solid rgba(128,128,128,.3);border-radius:8px;padding:10px 12px}
.tx .big{font-size:26px;font-weight:600}
.tx .scroll{max-height:540px;overflow:auto}
.tx .tiles{display:grid;gap:3px}
.tx .tiles img{width:100%;border:1px solid rgba(128,128,128,.4)}
.tx pre{white-space:pre-wrap;word-break:break-word;font:11.5px ui-monospace,Menlo,monospace;margin:0}
</style>"""

STATUSES = ("match", "differs", "extra", "missing")
STATUS_TEXT = {
    "match": "same letters and digits as the label",
    "differs": "same field, a letter or digit differs",
    "extra": "not in the label",
    "missing": "in the label, left out",
}


def esc(x) -> str:
    return html.escape(str(x))


def norm(value: str) -> str:
    """The lenient form of one value (casefolded, letters and digits only), as in metrics.lenient."""
    return next(iter(lenient(Counter({("", value): 1}))))[1]


def marked(a: str, b: str) -> str:
    """a as HTML with the characters that differ from b highlighted."""
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        out.append(esc(a[i1:i2]) if op == "equal" else f"<mark>{esc(a[i1:i2])}</mark>")
    return "".join(out)


def field_rows(record: dict, label: dict) -> list[tuple[str, str, str, str]]:
    """(status, path, model value, label value) for every value. Per field path, values with the same lenient form
    match; the rest are paired by text similarity (differs) when similar enough, else listed as extra or missing."""
    model_vals, label_vals = defaultdict(list), defaultdict(list)
    for (path, value), n in pairs(record).items():
        model_vals[path] += [value] * n
    for (path, value), n in pairs(label).items():
        label_vals[path] += [value] * n
    rows = []
    for path in list(label_vals) + [p for p in model_vals if p not in label_vals]:
        m, g = list(model_vals[path]), list(label_vals[path])
        for v in list(g):
            hit = next((x for x in m if norm(x) == norm(v)), None)
            if hit is not None:
                m.remove(hit)
                g.remove(v)
                rows.append(("match", path, hit, v))
        while m and g:
            score, x, v = max((difflib.SequenceMatcher(None, x, v).ratio(), x, v) for x in m for v in g)
            if score < 0.6:
                break
            m.remove(x)
            g.remove(v)
            rows.append(("differs", path, x, v))
        rows += [("extra", path, x, "") for x in m] + [("missing", path, "", v) for v in g]
    return rows


def image_uri(im: Image.Image, width: int) -> str:
    im = im.copy()
    im.thumbnail((width, width * 10))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Explorer:
    def __init__(self, model_dir, train_rows: list[dict], eval_rows: list[dict], max_new_tokens: int = 1500):
        self.model, self.processor, self.device = load_model(str(model_dir))
        self.patches = TRAINED_PATCHES[family_name(self.processor)]
        self.pages = {f"Unseen · {r['cv_id']}": r for r in eval_rows} | {
            f"Trained on · {r['cv_id']}": r for r in train_rows
        }
        self.pick = widgets.Dropdown(options=list(self.pages), description="Page", layout=widgets.Layout(width="380px"))
        self.upload = widgets.FileUpload(accept=".png,.jpg,.jpeg,.pdf", multiple=False, description="Upload")
        self.tokens = widgets.IntSlider(
            value=max_new_tokens, min=200, max=3000, step=100, description="Max tokens", continuous_update=False
        )
        self.button = widgets.Button(description="Read page", button_style="primary", icon="play")
        self.status = widgets.HTML()
        self.picture = widgets.HTML(layout=widgets.Layout(width="380px", min_width="380px"))
        self.filter = widgets.ToggleButtons(options=["all", *STATUSES], value="all", style={"button_width": "auto"})
        self.fields, self.scores, self.answer, self.inputs = (widgets.HTML() for _ in range(4))
        self.tabs = widgets.Tab([widgets.VBox([self.filter, self.fields]), self.scores, self.answer, self.inputs])
        for i, name in enumerate(["Fields", "Scores", "JSON", "Model input"]):
            self.tabs.set_title(i, name)
        self.tabs.layout = widgets.Layout(flex="1 1 auto", min_width="420px")
        self.button.on_click(lambda _: self.read())
        self.filter.observe(lambda _: self.show_fields(), names="value")
        self.rows: list = []
        self.labelled = True
        self.box = widgets.VBox(
            [
                widgets.HTML(
                    CSS + f"<div class='tx'><h4>Page explorer</h4><span class='muted'>model {esc(model_dir)} on "
                    f"{self.device}, {self.patches} image tiles</span></div>"
                ),
                widgets.HBox([self.pick, self.upload]),
                widgets.HBox([self.tokens, self.button, self.status]),
                widgets.HBox([self.picture, self.tabs], layout=widgets.Layout(align_items="flex-start")),
            ]
        )

    def show(self) -> None:
        display(self.box)
        self.read()

    def uploaded(self) -> tuple[str, Image.Image] | None:
        """(file name, first page) of the uploaded file; ipywidgets 7 keeps uploads in a dict, 8 in a tuple."""
        v = self.upload.value
        if not v:
            return None
        f = v[0] if isinstance(v, (tuple, list)) else next(iter(v.values()))
        name = f.get("name") or f["metadata"]["name"]
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as tmp:
            tmp.write(bytes(f["content"]))
            tmp.flush()
            return name, page_images(Path(tmp.name))[0]

    def read(self) -> None:
        self.button.disabled = True
        self.status.value = "<span class='tx muted'>reading the page…</span>"
        try:
            up = self.uploaded()
            if up:
                title, image, label = up[0], up[1], None
            else:
                row = self.pages[self.pick.value]
                title, image, label = self.pick.value, page_image(row), row["record"]
            t0 = time.time()
            answer, n_tokens = generate_pages(self.model, self.processor, [image], self.patches, self.tokens.value)[0]
            seconds = time.time() - t0
            record = read_answer(answer, fix=True)[0]
            self.render(title, image, label, answer, record, n_tokens, seconds)
            self.status.value = f"<span class='tx muted'>{n_tokens} tokens in {seconds:.1f} s</span>"
        except Exception as e:  # shown in the panel, then raised so the cell output has the traceback
            self.status.value = f"<span class='tx' style='color:#cf222e'>{esc(type(e).__name__)}: {esc(e)}</span>"
            raise
        finally:
            self.button.disabled = False

    def render(self, title, image, label, answer, record, n_tokens, seconds) -> None:
        self.picture.value = (
            f"<div class='tx'><img src='{image_uri(image, 760)}' style='width:370px;border:1px solid "
            f"rgba(128,128,128,.4)'><div class='muted'>{esc(title)} · {image.width}×{image.height} px</div></div>"
        )
        try:
            json.loads(answer)
            state = "valid JSON"
        except json.JSONDecodeError:
            state = "broken JSON, repaired" if record else "broken JSON, nothing could be repaired"
        hit_limit = n_tokens >= self.tokens.value
        self.answer.value = (
            f"<div class='tx scroll'><p><b>{state}</b>{' · stopped at the token limit' if hit_limit else ''}. "
            "The record below is what <code>tiny-resume-parser parse</code> returns: a broken answer keeps its "
            "complete values up to the break.</p><pre>"
            + esc(json.dumps(record, ensure_ascii=False, indent=1))
            + "</pre></div>"
        )
        self.labelled = label is not None
        self.rows = field_rows(record, label) if self.labelled else [("extra", p, v, "") for p, v in pairs(record)]
        self.show_fields()
        self.show_scores(record, label, state, n_tokens, seconds, hit_limit)
        self.show_inputs(image, n_tokens)

    def show_fields(self) -> None:
        counts = Counter(s for s, *_ in self.rows)
        want = self.filter.value
        body = "".join(
            f"<tr><td><span class='badge {s}' title='{STATUS_TEXT[s]}'>{s}</span></td><td><code>{esc(p)}</code></td>"
            f"<td>{marked(m, g) if s == 'differs' else esc(m)}</td><td>{marked(g, m) if s == 'differs' else esc(g)}</td></tr>"
            for s, p, m, g in self.rows
            if want == "all" or s == want
        )
        tally = " · ".join(f"<span class='badge {s}'>{counts[s]} {s}</span>" for s in STATUSES)
        note = (
            f"{tally}<br>Each row is one (field, value) pair. Field F1 counts a model value as correct when the label "
            "has the same value in the same field; list order does not matter. Highlighted characters differ."
            if self.labelled
            else "An uploaded page has no label, so every value is listed without a status."
        )
        self.fields.value = (
            f"<div class='tx'><p class='muted'>{note}</p><div class='scroll'><table><tr><th></th><th>field</th>"
            f"<th>model</th><th>label</th></tr>{body}</table></div></div>"
        )

    def show_scores(self, record, label, state, n_tokens, seconds, hit_limit) -> None:
        facts = f"{state} · {n_tokens} answer tokens{' (limit reached)' if hit_limit else ''} · {seconds:.1f} s"
        if label is None:
            self.scores.value = f"<div class='tx'><p>{facts}</p><p class='muted'>No label: nothing to score.</p></div>"
            return
        pred, gold = pairs(record), pairs(label)
        cards = []
        for name, p, g, text in [
            ("strict", pred, gold, "The value must equal the label value exactly."),
            (
                "lenient",
                lenient(pred),
                lenient(gold),
                "Case, spaces and punctuation are ignored; letters and digits must match. The main score.",
            ),
            (
                "content",
                content(pred),
                content(gold),
                "The words of each section, wherever they were placed: splitting or moving text is not an error.",
            ),
        ]:
            hit = sum((p & g).values())
            pr, rc, f = f1(hit, sum(p.values()), sum(g.values()))
            cards.append(
                f"<div class='card'><div class='muted'>{name} F1</div><div class='big'>{f:.2f}</div>"
                f"<div>precision {pr:.2f} · recall {rc:.2f}</div><div class='muted'>{hit} of {sum(g.values())} "
                f"label items found</div><p class='muted' style='margin:6px 0 0'>{text}</p></div>"
            )
        self.scores.value = (
            f"<div class='tx'><p>{facts}</p><div class='cards'>{''.join(cards)}</div><p class='muted'>Precision: share "
            "of the model's values that are in the label. Recall: share of the label's values the model wrote. F1: "
            "their harmonic mean. The full dataset scores pool all pages before dividing.</p></div>"
        )

    def show_inputs(self, image, n_tokens) -> None:
        batch = self.processor(
            text=[prompt_text(self.processor)],
            images=[image],
            return_tensors="pt",
            **family(self.processor)["size"](self.patches),
        )
        pv = batch["pixel_values"].reshape(-1, *batch["pixel_values"].shape[-3:]).float()
        ip = self.processor.image_processor
        std, mean = pv.new_tensor(ip.image_std)[:, None, None], pv.new_tensor(ip.image_mean)[:, None, None]
        tiles = [
            Image.fromarray(((t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")) for t in pv
        ]
        n_img = int((batch["input_ids"] == self.processor.image_token_id).sum())
        n_prompt = batch["input_ids"].shape[1] - n_img
        *grid, whole = tiles
        w, h = image.size
        scale = ip.size["longest_edge"] / max(w, h) if "longest_edge" in ip.size else 1
        cols = max(1, math.ceil(w * min(scale, 1) / pv.shape[-1]))
        cols = cols if len(grid) % cols == 0 else len(grid)
        cells = "".join(f"<img src='{image_uri(t, 120)}'>" for t in grid)
        self.inputs.value = (
            f"<div class='tx'><p>The page is scaled and cut into <b>{len(grid)} tiles</b> of {pv.shape[-1]} px, plus "
            f"<b>one copy of the whole page</b> at the same size. Each image becomes "
            f"{n_img // len(tiles)} tokens, so the model reads <b>{n_img} image tokens</b> and <b>{n_prompt} prompt "
            f"tokens</b>, then writes <b>{n_tokens} answer tokens</b> one at a time.</p>"
            f"<div style='display:flex;gap:14px;align-items:flex-start'><div class='tiles' style='grid-template-columns:"
            f"repeat({cols},1fr);width:{cols * 92}px'>{cells}</div><div><img src='{image_uri(whole, 120)}' "
            "style='width:120px;border:1px solid rgba(128,128,128,.4)'><div class='muted'>whole page</div></div></div>"
            "<p class='muted'>Small text survives only in the tiles; the whole-page copy gives the layout. More tiles "
            "means more detail and more image tokens.</p></div>"
        )


def report(explorer: Explorer) -> None:
    """The current panel as plain HTML, for saved copies of the notebook where widgets do not render."""
    display(HTML(CSS + "".join(f"<div class='tx'>{w.value}</div>" for w in [explorer.scores, explorer.fields])))
