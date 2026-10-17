"""Inputs a user or a training job hands to the package: PDF pages and training config files."""

import tomllib
from pathlib import Path

from tiny_resume_parser.parse import PDF_DPI, page_images
from tiny_resume_parser.train import build_parser

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))


def test_pdf_pages_render_at_training_resolution(tmp_path):
    # Training pages are A4 at ~120 dpi (991 x 1404 px); a PDF rendered at another size would shift the input.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    for _ in range(2):
        doc.new_page(8.27 * 72, 11.69 * 72)  # A4 in PDF points
    pdf = tmp_path / "cv.pdf"
    doc.save(pdf)
    images = page_images(pdf)
    assert len(images) == 2 and all(im.mode == "RGB" for im in images)
    assert abs(images[0].size[0] - 8.27 * PDF_DPI) <= 2 and abs(images[0].size[1] - 11.69 * PDF_DPI) <= 2


def test_every_config_key_is_a_training_option():
    # A misspelt key would otherwise be ignored and the run would silently use a default.
    options = {a.dest for a in build_parser()._actions}
    assert CONFIGS, "no configs/*.toml found"
    for path in CONFIGS:
        unknown = set(tomllib.loads(path.read_text())) - options
        assert not unknown, f"{path.name}: {sorted(unknown)}"


def test_parse_command_flags_unreadable_pages(tmp_path, monkeypatch):
    # read_cv end to end with the model replaced by stored answers: a cut-off answer keeps its complete values and
    # counts as valid, a page with no JSON is reported as not valid instead of as an empty CV.
    from PIL import Image

    import tiny_resume_parser.parse as p

    answers = iter(['{"basics": {"name": "Jean Picard", "emails": ["jean@exa', "I cannot read this page."])
    monkeypatch.setattr(p, "load_model", lambda model, device=None: (None, None, "cpu"))
    monkeypatch.setattr(p, "generate_pages", lambda *a, **k: [(next(answers), 0)])
    files = [tmp_path / "a.png", tmp_path / "b.png"]
    for f in files:
        Image.new("RGB", (8, 8), "white").save(f)
    first, second = p.read_cv(files, "any-model", max_patches=12)
    assert first["valid"] and first["record"]["basics"]["name"] == "Jean Picard"
    assert not second["valid"] and second["record"] == {}
