"""CV field schema and DocTags ↔ flat-dict conversion.

This module encodes a flat key-value CV record (the format produced by the VLM
auto-labeler and consumed by downstream consumers) into the DocTags markup that
SmolDocling is trained to emit, and decodes a generated DocTags string back into
a flat dict.

The encoding scheme is the result of the M1 round-trip prototype
(`data/processed` development history). The critical finding:

    DocTags reliably round-trips `section_header` + `list_item` pairs, but
    DROPS standalone `paragraph` items. Therefore *every* field value — scalar
    or list — is encoded as one or more `list_item`s under a `section_header`
    named after the field. Scalars are recovered by taking the single element;
    lists stay lists.

This keeps the model's target format uniform and the projection lossless for
all 17 fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from docling_core.types.doc import (
    DocItemLabel,
    DocTagsDocument,
    DoclingDocument,
    ProvenanceItem,
)
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DocTagsPage

# --- Field schema ---------------------------------------------------------

# Scalar fields: model outputs a single string; encoded as a 1-element list and
# projected back to str.
SCALAR_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "email",
        "phone",
        "address",
        "birthdate",
        "profile",
        "github",
        "linkedin",
    }
)

# List fields: model outputs a list of strings.
LIST_FIELDS: frozenset[str] = frozenset(
    {
        "skills",
        "languages",
        "interests",
        "experience",
        "education",
        "projects",
        "certifications",
    }
)

# Meta fields: derived from the document (not extracted from content), stored as
# scalars. `photo_present` is bool-as-string, `num_pages` is int-as-string.
META_FIELDS: frozenset[str] = frozenset({"photo_present", "num_pages"})

# The full ordered schema. Order is fixed so DocTags targets are deterministic.
FIELDS: tuple[str, ...] = (
    "name",
    "email",
    "phone",
    "address",
    "birthdate",
    "profile",
    "github",
    "linkedin",
    "skills",
    "languages",
    "interests",
    "experience",
    "education",
    "projects",
    "certifications",
    "photo_present",
    "num_pages",
)

# Fields that are always treated as scalars on projection (single-element list -> str).
_SCALAR_LIKE = SCALAR_FIELDS | META_FIELDS


@dataclass(frozen=True)
class CVRecord:
    """A typed view of a flat CV record. All fields optional; absent -> default."""

    name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    birthdate: str = ""
    profile: str = ""
    github: str = ""
    linkedin: str = ""
    skills: list[str] | None = None
    languages: list[str] | None = None
    interests: list[str] | None = None
    experience: list[str] | None = None
    education: list[str] | None = None
    projects: list[str] | None = None
    certifications: list[str] | None = None
    photo_present: bool = False
    num_pages: int = 1


def empty_record() -> dict:
    """Return a complete flat record with empty defaults (no nulls)."""
    rec: dict = {f: "" for f in SCALAR_FIELDS}
    rec.update({f: [] for f in LIST_FIELDS})
    rec["photo_present"] = False
    rec["num_pages"] = 1
    return rec


def normalize(record: dict) -> dict:
    """Coerce a record to the full schema with correct types and no nulls.

    Used both on VLM output (defensive) and on decoded DocTags output.
    """
    out = empty_record()
    for f in FIELDS:
        if f not in record:
            continue
        val = record[f]
        if val is None:
            continue
        if f in LIST_FIELDS:
            if isinstance(val, str):
                val = [val] if val else []
            out[f] = [str(v) for v in val]
        elif f == "photo_present":
            if isinstance(val, bool):
                out[f] = val
            else:
                out[f] = str(val).strip().lower() in {"true", "1", "yes"}
        elif f == "num_pages":
            try:
                out[f] = int(val)
            except (TypeError, ValueError):
                out[f] = 1
        else:  # scalar string
            out[f] = "" if isinstance(val, list) and not val else str(val)
    return out


# --- Encoding: flat dict -> DoclingDocument -> DocTags string -------------


# --- COCO class -> schema field mapping ----------------------------------
# The invoicemodelisation dataset's 18 COCO classes map to our schema as below.
# `Resume` (the page banner region) and `words-pPx7` (junk) are NOT schema
# fields and are dropped. `Image` -> photo_present (presence-as-bool). Verified
# against the actual _annotations.coco.json category names.
COCO_CLASS_TO_FIELD: dict[str, str] = {
    "Name": "name",
    "Email": "email",
    "Phone": "phone",
    "Adresse": "address",
    "Github": "github",
    "LinkedIn": "linkedin",
    "Profil": "profile",
    "Naissance": "birthdate",
    "Experience": "experience",
    "Education": "education",
    "Skills": "skills",
    "Projects": "projects",
    "Certifications": "certifications",
    "Languages": "languages",
    "Interests": "interests",
    "Image": "photo_present",  # bbox presence -> photo_present=True
}
# COCO classes that carry no schema meaning.
COCO_JUNK_CLASSES: frozenset[str] = frozenset({"words-pPx7", "Resume"})


def _bbox_xywh_to_pixels(
    bbox_xywh: tuple[float, float, float, float],
    page_size: tuple[int, int],
) -> BoundingBox:
    """COCO (x, y, w, h) in absolute pixels -> pixel-space BoundingBox.

    IMPORTANT coordinate-space contract (verified empirically):
    - COCO bboxes are ABSOLUTE PIXELS on the page canvas (range 0..640 for our
      640×640 Roboflow images; never normalized).
    - Docling's `get_location()` does `bbox / page_size` itself, so the
      BoundingBox MUST store pixel values, NOT pre-normalized [0,1] values.
      Storing normalized values yields `loc_0` tokens (the original bug).
    - Our images are letterboxed (not squished) into 640×640, so pixel coords
      map 1:1 to what the model sees — no aspect-ratio rescaling needed.
    """
    x, y, w, h = bbox_xywh
    return BoundingBox(l=x, t=y, r=x + w, b=y + h)


def record_to_doctags(
    record: dict,
    page_size: tuple[int, int] = (640, 640),
    field_bboxes: dict[str, tuple[float, float, float, float]] | None = None,
) -> str:
    """Encode a normalized CV record as a DocTags string.

    Args:
        record: flat dict (will be normalized first).
        page_size: (width, height) of the page canvas in pixels. Defaults to
            the Roboflow 640×640 canvas. Used to declare the Docling page and
            as the reference space for bbox coordinates.
        field_bboxes: optional mapping field -> (x, y, w, h) ABSOLUTE-PIXEL
            bbox taken from the COCO annotation. Seeds accurate `loc` tokens
            (the spatial part of the target). Fields without a bbox fall back
            to a full-page box so the loc token is non-degenerate.

    Returns:
        DocTags markup string, ready to be a training target.
    """
    rec = normalize(record)
    field_bboxes = field_bboxes or {}
    w, h = page_size

    doc = DoclingDocument(name="cv")
    doc.add_page(page_no=1, size=Size(width=w, height=h))

    for field in FIELDS:
        bbox_xywh = field_bboxes.get(field)
        if bbox_xywh is None:
            # No COCO box for this field: use full-page box (non-degenerate loc).
            bbox_xywh = (0.0, 0.0, float(w), float(h))
        bb = _bbox_xywh_to_pixels(bbox_xywh, page_size)
        prov = ProvenanceItem(page_no=1, bbox=bb, charspan=(0, 0))

        # Field name as section header (always survives the round-trip).
        doc.add_heading(text=field, prov=prov)

        # Value(s) as list_item(s) — the only content node DocTags reliably
        # preserves. Scalars become a single-element list.
        if field in LIST_FIELDS:
            values = rec[field]
        elif field == "photo_present":
            values = ["true" if rec[field] else "false"]
        elif field == "num_pages":
            values = [str(rec[field])]
        else:
            values = [rec[field]] if rec[field] != "" else []

        grp = doc.add_group(label=None, name=f"{field}_list")
        for v in values:
            doc.add_list_item(text=str(v), prov=prov, parent=grp)

    return doc.export_to_doctags()


# --- Decoding: DocTags string -> flat dict -------------------------------


def doctags_to_record(doctags: str) -> dict:
    """Decode a DocTags string back into a normalized flat CV record.

    Inverse of `record_to_doctags`. Walks the recovered document's text items
    in body order: each `section_header` opens a field, subsequent `list_item`s
    populate it until the next header.
    """
    dt_doc = DocTagsDocument(pages=[DocTagsPage(tokens=doctags)])
    doc = DoclingDocument.load_from_doctags(dt_doc)

    raw: dict[str, list[str]] = {}
    current: str | None = None
    values: list[str] = []

    def flush() -> None:
        nonlocal current, values
        if current is not None:
            raw[current] = values
            values = []

    for item in doc.texts:
        if item.label == "section_header" and item.text:
            flush()
            current = item.text
        elif item.label == "list_item" and item.text and current:
            values.append(item.text)
    flush()

    # Project raw {field: [values]} onto the typed schema.
    rec = empty_record()
    for field in FIELDS:
        vals = raw.get(field, [])
        if field in LIST_FIELDS:
            rec[field] = vals
        elif field == "photo_present":
            rec[field] = any(v.strip().lower() == "true" for v in vals)
        elif field == "num_pages":
            try:
                rec[field] = int(vals[0]) if vals else 1
            except ValueError:
                rec[field] = 1
        else:  # scalar
            rec[field] = vals[0] if vals else ""
    return rec


# --- Round-trip self-check -----------------------------------------------


def coco_annotations_to_field_bboxes(
    coco_boxes: list[tuple[str, tuple[float, float, float, float]]],
) -> tuple[dict[str, tuple[float, float, float, float]], bool]:
    """Convert raw COCO (class_name, bbox_xywh) pairs into the field->bbox map
    that `record_to_doctags` consumes.

    Policy for the multi-box case (verified against the dataset: list fields
    like Experience commonly have several boxes, one per job entry):
    - LIST_FIELDS: keep ALL boxes — but `record_to_doctags` currently takes one
      bbox per field. For the spatial loc token we use the ENCLOSING box (union)
      so the loc token spans the whole field region; individual list entries
      inherit the same loc. (Per-entry bboxes would require a per-item prov,
      a future enhancement.)
    - SCALAR_FIELDS / META_FIELDS: take the TOPMOST box (smallest y), the usual
      reading-order convention.
    - `Image` (photo) class: bbox presence -> photo_present; the bbox itself
      locates the photo.

    Args:
        coco_boxes: [(class_name, (x,y,w,h)), ...] straight from COCO, pixel space.

    Returns:
        (field_bboxes, photo_detected) where field_bboxes maps schema field ->
        enclosing/topmost (x,y,w,h) and photo_detected is True if an `Image`
        class box was present.
    """
    by_field: dict[str, list[tuple[float, float, float, float]]] = {}
    photo_detected = False
    for cls, bbox in coco_boxes:
        if cls in COCO_JUNK_CLASSES:
            continue
        field = COCO_CLASS_TO_FIELD.get(cls)
        if field is None:
            continue
        if field == "photo_present":
            photo_detected = True
            by_field.setdefault("photo_present", []).append(bbox)
            continue
        by_field.setdefault(field, []).append(bbox)

    field_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for field, boxes in by_field.items():
        if field in LIST_FIELDS:
            # Enclosing (union) box for the loc token.
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            field_bboxes[field] = (x0, y0, x1 - x0, y1 - y0)
        else:
            # Topmost box (smallest y).
            field_bboxes[field] = min(boxes, key=lambda b: b[1])
    return field_bboxes, photo_detected


def selftest() -> bool:
    """Verify encode(decode(x)) == normalize(x) for a full sample record, AND
    that real pixel bboxes produce non-zero loc tokens.

    Run this to confirm the DocTags↔dict mapping is still lossless and the
    coordinate-space handling is correct after any change to this module.
    Used as the M1 gate.
    """
    sample = {
        "name": "Jane Smith",
        "email": "jane@x.com",
        "phone": "+44 20 7946",
        "address": "London",
        "birthdate": "1990-05-12",
        "profile": "Data scientist",
        "github": "github.com/jane",
        "linkedin": "linkedin.com/in/jane",
        "skills": ["Python", "SQL"],
        "languages": ["English"],
        "interests": ["Chess"],
        "experience": ["DS at Beta"],
        "education": ["PhD Oxford"],
        "projects": ["Dashboard"],
        "certifications": ["AWS ML"],
        "photo_present": True,
        "num_pages": 1,
    }
    expected = normalize(sample)

    # 1) round-trip WITHOUT bboxes (content fidelity)
    got = doctags_to_record(record_to_doctags(sample))
    content_ok = got == expected
    if not content_ok:
        for f in FIELDS:
            if got.get(f) != expected.get(f):
                print(f"  CONTENT MISMATCH {f}: expected={expected.get(f)!r} got={got.get(f)!r}")

    # 2) round-trip WITH real pixel bboxes (loc-token fidelity)
    bboxes = {
        "name": (38, 226, 562, 88),
        "email": (38, 320, 300, 30),
        "phone": (340, 320, 260, 30),
    }
    dt = record_to_doctags(sample, field_bboxes=bboxes)
    # Must contain non-zero loc tokens (not all loc_0).
    loc_zero_only = dt.count("<loc_0>") > 0 and "<loc_" not in dt.replace("<loc_0>", "")
    # Check at least one loc token > 0 exists
    import re
    loc_vals = [int(m) for m in re.findall(r"<loc_(\d+)>", dt)]
    loc_ok = any(v > 0 for v in loc_vals)
    if not loc_ok:
        print(f"  LOC MISMATCH: all loc tokens are 0. doctags={dt[:200]}")

    # 3) content still round-trips with bboxes present
    got2 = doctags_to_record(dt)
    bbox_content_ok = got2 == expected

    return content_ok and loc_ok and bbox_content_ok


if __name__ == "__main__":
    import sys

    ok = selftest()
    print(f"schema self-test: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
