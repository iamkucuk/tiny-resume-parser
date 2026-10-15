# CV / Resume Datasets — Research Findings

Goal: find **real** CV datasets composed of **PDFs and images** (ideally with parsing
ground-truth) that can be used to build/test a CV parser.

## ✅ DECISION (final)

After downloading, deduplicating, and verifying all candidates, **two datasets
were selected** and are on disk. All others were rejected and removed.

| Role | Dataset | Location | What it is |
|---|---|---|---|
| **Primary (labeled)** | `invoicemodelisation/cv-parsing` original uploads (fork `vizioai/cv-parsing-ksgbu-qiqkv` v1) | `data/roboflow_source/` | 2,162 CVs / 2,451 pages after dedup (202 multi-page), CC BY 4.0. The v8 export in `data/roboflow/` is augmented (3 copies per train page) — do not train on it |
| **Secondary (unlabeled scale)** | `hadikp/resume-data-pdf` | `data/raw/` | 8,905 single-page resume images, no labels, MIT |

**Dropped (reasons):**
- `intelicruit/resume-parser` v3 — only 1,264 unique CVs after dedup, 1 multi-page CV, 8.3× augmentation inflation. Strictly dominated by invoicemodelisation. *(was 2.1 GB)*
- `seif/resume_section_detection` — export artifacts expired from Roboflow storage, undownloadable.
- `snehaanbhawal`, `suriyaganesh` (Kaggle LiveCareer sets) — excluded per user direction.

Pages grouped into CVs: `data/processed/documents.jsonl` (see `PLAN.md`).

---

> ⚠️ **Licensing heads-up (read first):** The largest "free" resume corpora
> (LiveCareer, Indeed) are scraped from commercial job sites. They are fine for
> **internal research / model training** in many jurisdictions, but they are
> **NOT safe to redistribute** as-is. If you need a dataset you can ship inside
> a product or republish, prefer the explicitly-licensed ones (Innovatiana,
> CVParserPro) or synthetic data.

---

## TL;DR — what actually exists

| Need | Best option |
|---|---|
| Real **PDF** files, categorized | `snehaanbhawal/resume-dataset` (2,400+ PDFs, LiveCareer) |
| Real **PDF** files, large | `suriyaganesh/resume-dataset-structured` (54k, OCR'd + structured) |
| Real **images** with bboxes | Roboflow `resume-parser` (4,493 imgs) / `resume-layout` |
| **NER ground-truth** (skills, edu, etc.) | DataTurks 220-resume set (gold standard) |
| Large NER set, modern | `Mehyaar/Annotated_NER_PDF_Resumes` (5,029 CVs) |
| **Commercially licensed** (safe to ship) | Innovatiana "Summary Dataset" / CVParserPro IT set |

---

## 1. PDF-based datasets (real PDF files)

### `snehaanbhawal/resume-dataset` (Kaggle) — ⭐ the canonical starting point
- **2,400+ resumes** in **both PDF and raw text/string** format
- PDFs organized into **24 category folders** (HR, IT, Sales, Healthcare, etc.)
- Source: **LiveCareer.com** scrape
- This is the "root" dataset that several others are derived from
- https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- License: LiveCareer ToS — research/internal use only

### `hadikp/resume-data-pdf` (Kaggle)
- **PDFs converted from the original resume-image Kaggle dataset** (so it traces
  back to LiveCareer images too)
- Categorized, ~same coverage as snehaanbhawal
- https://www.kaggle.com/datasets/hadikp/resume-data-pdf

### `suriyaganesh/resume-dataset-structured` (Kaggle) — largest
- **~54,000 resumes** structured via OCR → LLM pipeline
- Good for scale; ground-truth quality is LLM-derived (not human-annotated)
- https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured

### CVParserPro IT Resume Dataset — clean & licensed
- **50+ realistic IT resumes in DOCX** (not PDF, but trivially convertible)
- Explicitly built for **testing parsers and ATS software**
- https://cvparserpro.io/it-resume-dataset

---

## 2. Image-based datasets (real images, often annotated)

### Roboflow Universe — best source for **images with bounding-box annotations**

**All numbers below are VERIFIED** — downloaded via API and counted from the
actual files (Aug 2026). The public web pages undercount because they show old
versions; the live latest versions are much bigger.

| Dataset (workspace/project) | Latest ver | Total images | Classes | License |
|---|---|---|---|---|
| `resume-parsing-mk6d6/resume-parser` | v1 | **10,803** | 15 fields | CC BY 4.0 |
| `intelicruit/resume-parser` | v3 | **10,529** | 15 fields | CC BY 4.0 |
| `seif/resume_section_detection` | v1/v2 | **8,891** | sections | — |
| **`invoicemodelisation/cv-parsing`** ⭐ | **v8** | **7,312** | **18 fields** | **CC BY 4.0** |
| `nadhir/resume-parsing` | v3 | 1,166 | resume-sections | — |
| `smartc/resume-parser` | v1 | 724 | fields | — |

`resume-parsing-mk6d6` and `intelicruit` appear to be the **same underlying
project** (identical classes, near-identical image counts, same `cv{N}` naming)
— pick one.

`invoicemodelisation/cv-parsing` v8 **classes (18):** Name, Email, Phone,
Adresse, Experience, Education, Skills, Certifications, Projects, Languages,
Interests, Github, LinkedIn, Image (photo), Profil, Naissance (DOB), Resume,
+ 1 junk class `words-pPx7` (filter out).

Export: **COCO / YOLO / VOC**. Downloaded locally to `data/roboflow/`.

#### ✅ Multi-page analysis (VERIFIED by filename inspection)

Definition used: a multi-page CV = one CV whose pages appear as **multiple
distinct page-images** linked by filename (e.g. `CV_Mohamed_page1`,
`CV_Mohamed_page2`). This is exactly the multi-page signal you asked about.

| Dataset | Distinct CVs | **Multi-page CVs** | Max pages | Notes |
|---|---|---|---|---|
| **`invoicemodelisation/cv-parsing` v8** ⭐ | ~2,180 | **204** | **9** | Rich `_page1`/`_page2`/`page-0001` naming; 18 classes annotated per page |
| `intelicruit/resume-parser` v3 | ~1,267 | **~11** | 3 | Mostly single-page CVs × heavy augmentation (each CV duplicated 6–22×) |
| `seif/resume_section_detection` | TBD | TBD | — | downloading (v2 export missing; retrying v1) |

**The clear winner for multi-page is `invoicemodelisation/cv-parsing` v8.**
- 204 genuine multi-page documents (153 are 2-page, 32 are 3-page, up to 9 pages).
- Pages are linked by explicit markers: `_page1`, `_page2`, `_page3`, `page-0001`, etc.
- Each page has its own bounding-box annotations across all 18 classes.
- ⚠️ Caveat: ~1,892 documents only show page 1 — could be genuinely single-page
  OR multi-page CVs whose later pages weren't uploaded. The 204 explicitly
  multi-page ones are the safe, confirmed subset.

#### How to reconstruct documents
Roboflow stores each page as a standalone image. To rebuild multi-page docs:
1. Parse the base filename, strip the `pageN` marker → document ID.
2. Group images by document ID, sort by page number.
3. Merge per-page annotations into one document record.

(There's no explicit parent/child link in the COCO file — the filename is the
only grouping signal. I verified this works for the 204 multi-page CVs above.)

### `noran-mohamed/Resume-Classification-Dataset` (GitHub)
- Resumes from **Google Images, Bing Images, and LiveCareer**
- Has a `Category` label per resume (classification, not bbox)
- https://github.com/noran-mohamed/Resume-Classification-Dataset

---

## 3. Ground-truth / annotated datasets (for training a parser)

These are the ones you want if you're training a model, not just collecting files.

### DataTurks — `Entity-Recognition-In-Resumes-SpaCy` — ⭐ gold standard
- **220 resumes** sourced from Indeed, **manually annotated** with NER
- **Entity types:** College Name, Degree, Graduation Year, Years of Experience,
  Companies worked at, Location, Designation, Skills
- GitHub: https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy
- Kaggle (mirror): https://www.kaggle.com/dataturks/resume-entities-for-ner
- Suggested split: 200 train / 20 test
- Format: JSON with character-offset annotations
- License: MIT (the annotation code) — resume content is research-use

### `Mehyaar/Annotated_NER_PDF_Resumes` (HuggingFace) — largest NER set
- **5,029 CVs** annotated with **IT skills** via NER (manually labeled)
- Good scale if your domain is IT/tech
- https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes

### `yashpwrr/resume-ner-training-dataset` (Kaggle) — merged & expanded
- **5,960 standardized resume samples** for NER
- Combines 4 sources (incl. DataTurks)
- https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset

### `AjNavneet/Resume-Parser-Spacy-NER` (GitHub)
- Ships **JSON files with labeled fields** ready for spaCy NER training
- Fields: Location, Designation, Name, Years of Experience, College, Degree,
  Graduation Year, Companies, Skills
- https://github.com/AjNavneet/Resume-Parser-Spacy-NER

### CareerCorpus (Mendeley Data / ScienceDirect)
- **302 annotated resumes** across **6 occupational categories**
- Sourced from LiveCareer (Kaggle) + LinkedIn
- Designed for NLP / IE tasks; has academic paper describing the annotation
- https://data.mendeley.com/datasets/wzzwn37gmd/1
- Paper: https://www.sciencedirect.com/science/article/pii/S2352340926001204

### `datasetmaster/resumes` (HuggingFace)
- Merged real + synthetic resumes **normalized to a common JSON schema**
- Useful as a "what good output looks like" reference for your parser
- https://huggingface.co/datasets/datasetmaster/resumes

---

## 4. Commercially-licensed / safe-to-redistribute

### Innovatiana — "Summary Dataset"
- **2,400+ resumes** in **text, HTML, and PDF** formats
- Each tagged with a professional category (24 categories)
- **Explicit commercial license available** (rare and important)
- https://www.innovatiana.com/en/datasets/resume-dataset

### CVParserPro IT Resume Dataset
- 50+ DOCX IT resumes, explicitly for parser/ATS testing
- https://cvparserpro.io/it-resume-dataset

---

## 5. Related — document-layout datasets (not resume-specific, but useful)

If you want a layout/segmentation model rather than just NER:

- **DocLayNet** — 80,863 human-annotated pages, COCO format, 11 layout classes.
  Resumes aren't a standalone class but it's the gold-standard layout methodology.
  https://github.com/DS4SD/DocLayNet · https://huggingface.co/datasets/docling-project/DocLayNet
- **HURIDOCS/pdf-document-layout-analysis** — model + service for PDF layout
  https://huggingface.co/HURIDOCS/pdf-document-layout-analysis
- **PubLayNet / DocBank** — large-scale general document layout (text-heavy)

---

## Recommended approach for your `cv_parser` project

1. **Start with** `snehaanbhawal/resume-dataset` (2,400 real PDFs) as your
   core corpus — it's the most widely-used and well-understood.
2. **Pair it with** the DataTurks 220-resume NER set as your labeled
   evaluation/training ground truth.
3. **For image/visual parsing** (OCR pipeline), pull Roboflow `resume-parser`
   (4,493 images, bboxes) + `resume-layout` for region detection.
4. **If you need scale**, add `Mehyaar/Annotated_NER_PDF_Resumes` (5,029 CVs).
5. **If you need to ship/redistribute**, license Innovatiana instead of using
   the LiveCareer-derived sets.
6. **If you need DOCX**, CVParserPro gives you a clean, parser-test-friendly set.

### Likely single best "PDFs + images + labels" combo
> `snehaanbhawal` (PDF corpus) + Roboflow `resume-parser` (images+bboxes) +
> DataTurks 220 (NER labels) → covers all three modalities with real data.
