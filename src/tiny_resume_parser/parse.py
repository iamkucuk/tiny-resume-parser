"""Read CV files (page images or PDFs) with a trained student and write one JSON record per page.

    tiny-resume-parser parse cv.pdf --model PATH_OR_HF_ID              # prints JSON
    tiny-resume-parser parse page1.png page2.jpg --model PATH --out cv.json

PDF pages are rendered to images at 120 dpi, the resolution of the training pages (A4 = 992 x 1403 px); the text
layer of the PDF is never read. Each page is read on its own; merging the pages of one CV into a single record is
not implemented yet. Answers are repaired before use (metrics.repair: a cut-off answer keeps its complete values).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .generate import generate_pages, load_model
from .metrics import read_answer
from .models import TRAINED_PATCHES, family_name

PDF_DPI = 120


def page_images(path: Path) -> list[Image.Image]:
    """All pages of one file as RGB images: a PDF gives one image per page, an image file gives itself."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        return [page.render(scale=PDF_DPI / 72).to_pil().convert("RGB") for page in pdf]
    return [Image.open(path).convert("RGB")]


def read_cv(
    paths: list[Path], model: str, device: str | None = None, max_patches: int | None = None, max_new_tokens: int = 3000
) -> list[dict]:
    """[{file, page, valid, record}] for every page of every file; valid = the record passes the schema."""
    m, processor, _ = load_model(model, device=device)
    max_patches = max_patches or TRAINED_PATCHES[family_name(processor)]
    out = []
    for path in paths:
        for i, image in enumerate(page_images(path), 1):
            answer, _ = generate_pages(m, processor, [image], max_patches, max_new_tokens)[0]
            record, status = read_answer(answer, fix=True)
            out.append({"file": str(path), "page": i, "valid": status == "valid", "record": record})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tiny-resume-parser parse", description=__doc__.splitlines()[0])
    ap.add_argument("files", type=Path, nargs="+", help="PDF or image files (PNG, JPEG, ...)")
    ap.add_argument("--model", required=True, help="trained model folder or Hugging Face id")
    ap.add_argument("--out", type=Path, help="write JSON here instead of printing it")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: cuda, else mps, else cpu")
    ap.add_argument(
        "--max-patches", type=int, help="image tiles per page; default: the value the model family was trained with"
    )
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    args = ap.parse_args(argv)
    pages = read_cv(args.files, args.model, args.device, args.max_patches, args.max_new_tokens)
    text = json.dumps(pages, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text)
        print(f"{len(pages)} pages -> {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
