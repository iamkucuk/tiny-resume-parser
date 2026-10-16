# A. Base models under 1B for CV page image -> JSON (research date 2026-10-16)

Method: parameter totals are the exact safetensors totals from `https://huggingface.co/api/models/<id>`; per-module splits were summed from the safetensors headers. This counts untied lm_head and the MTP head. Our ResuVL-L baseline (InternVL3-1B-hf) counts as 938,193,024 the same way (https://huggingface.co/OpenGVLab/InternVL3-1B-hf). Benchmark numbers are vendor-reported unless noted, and none of them test CV pages.

## Key findings
1. The field moved a lot in 2026. Four OCR-specialised models are under 1B, ship in transformers, and are Apache-2.0. Three of them (LightOnOCR-3-0.8B, OvisOCR2, Xiaomi-OCR-0) share one architecture, Qwen3.5-0.8B. One new model family in our code therefore opens three OCR post-trainings plus the generic base.
2. Non-embedding LM size, our strongest lever (2B > 1B), ranks as follows:
   - Qwen3.5-0.8B: 498M
   - Qwen3-0.6B: 440M
   - InternVL3-1B (Qwen2.5-0.5B): 358M
   - ERNIE-4.5-0.3B (PaddleOCR-VL): 255M
   - Granite-165M (granite-docling): 165M
3. PaddleOCR-VL has the strongest reader but the smallest LM of the serious options. That is a risk for assigning values to JSON fields.

## Candidates under 1B (total params from the HF API)

| Model | Total | Vision + LM | French / languages | Key numbers (source) | Licence | transformers | Output training |
|---|---|---|---|---|---|---|---|
| [LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) (2026-10-07) | 852,985,920 | ViT 100.6M + Qwen3.5-0.8B LM 752M (embeddings tied) | French-focused data; 20+ languages ([blog](https://huggingface.co/blog/lightonai/lightonocr-3)) | olmOCR-Bench 85.5; ParseBench 74.6; **French fr-bench-pdf2md 70.5** (Surya-2: 54.1, LightOnOCR-3-1B: 69.6) ([blog](https://huggingface.co/blog/lightonai/lightonocr-3)) | Apache-2.0 | `Qwen3_5ForConditionalGeneration`, needs >=5.5.4 | markdown; `grounding` mode adds labelled boxes; trained with SFT then GRPO |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) (2026-07-13) | 852,985,920 | Qwen3.5-0.8B post-trained, same shapes | not stated | OmniDocBench v1.6 96.58; PureDocBench Avg3 75.06 ([card](https://huggingface.co/ATH-MaaS/OvisOCR2), [report](https://arxiv.org/abs/2607.13639)) | Apache-2.0 | qwen3_5 | page to markdown, no layout detector; SFT + RL + on-policy distillation |
| [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) (2026-09-29) | 873,438,784 (853M without the 20.5M MTP head) | Qwen3.5-0.8B-Base post-trained on ~170M samples | not stated | OmniDocBench v1.6 96.83; OCR-VQA mean 83.2 (DocVQA 93.1, OCRBench 84.6); **in-house JSON key-info extraction 59.09 %** ([card](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0), [report](https://arxiv.org/abs/2609.36136)) | Apache-2.0 | qwen3_5 (transformers 5.17 pinned in card) | markdown plus a prompted "extract key information" to JSON-by-schema task |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (2026-02-28) | 873,438,784 (853M without MTP) | ViT 99M (12 layers, patch 16) + hybrid Gated-DeltaNet LM: 498M non-embedding + 254M embeddings | 201 languages ([card](https://huggingface.co/Qwen/Qwen3.5-0.8B)) | non-thinking OCRBench 79.1, CC-OCR 66.7, OmniDocBench1.5 70.6 (card: "Thinking / Non-thinking") | Apache-2.0 | native | general; use as the control to show whether OCR post-training helps |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) (also 1.5 and 1.0, same size) | 958,588,736 | NaViT-style ViT 466M (Keye-VL init) + MLP + ERNIE-4.5-0.3B (255M layers; 106M embeddings + 106M untied head) | **French listed explicitly** among 109 languages (Appendix B of [paper](https://arxiv.org/abs/2510.14528)); a French page is shown | OmniDocBench v1.6 96.33 (card); MDPBench overall 75.0, Latin 78.0 (1.5: 78.3 / 81.2) ([MonkeyOCRv2 paper Table 6](https://arxiv.org/abs/2607.11562)); olmOCR 80.0±1.0 for v1.0 ([paper](https://arxiv.org/abs/2510.14528)); Latin line edit distance 0.013 | Apache-2.0 | `paddleocr_vl` in transformers >=5 ([docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/paddleocr_vl.md)); HF example is element-level only | max_pixels 1280x28x28 (about 1.0 MP, <=1280 tokens) in stage 1, 2048x28x28 in stage 2; trained on text lines, blocks and pages; outputs OCR text, tables and charts; page parsing in the official stack uses a separate layout detector, which is not allowed here |
| [MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) (2026-07) | 877,939,456 | ViT-B 126M (frozen in their training) + Qwen3-0.6B (440M layers + 2x155.6M embed/head) | **17 languages incl. French**; MDPBench French 78.4 | MDPBench overall 83.3, Latin 84.2, #1 open model ([card](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing)); the authors' own benchmark line, so treat with care | Apache-2.0; dataset MonkeyDocv2 also Apache-2.0 per its [HF tags](https://huggingface.co/api/datasets/zenosai/MonkeyDocv2) | custom class, needs trust_remote_code | layout then crop recognition, so page-level reading needs adapting |
| [Surya-OCR-2](https://huggingface.co/datalab-to/surya-ocr-2) | 686,154,304 | Qwen3.5-style, 65k pruned vocab, 24 layers | French 89.3 % on their internal set | olmOCR-bench 83.3 % (card); French fr-bench-pdf2md only 54.1 (LightOn blog) | modified OpenRAIL-M: free for research, personal use, startups under $5M; **likely conflicts with our CC BY-NC release, so check before use** | qwen3_5 | HTML blocks with boxes |
| [Nemotron-Parse-2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) | 903,049,225 | C-RADIO ViT-H 652M + 10-layer mBART decoder (72k vocab) | no French evidence; gains reported on CJK and Indic | ParseBench 0.639, MOSCAR 0.910 (card) | OpenMDW-1.1 | remote code | markdown + boxes + classes; recommended input up to 1664x2048 |
| [Falcon-OCR 1.5](https://huggingface.co/tiiuae/Falcon-OCR) | 269,944,416 | one early-fusion transformer, no separate ViT | not documented | olmOCR 82.4, OmniDocBench e2e 86.47 (card; beats paddleocr-vl-1.6 78.7 in the same table) | Apache-2.0 | remote code | text / LaTeX / HTML; GRPO with omission and loop penalties |
| [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) | 448,718,848 | SigLIP2 86M + LFM2.5-350M | French listed | OCRBench 684, InfoVQA 43.0; card says "not well-suited for ... fine-grained OCR" | LFM Open License v1.0 ($10M revenue cap) | lfm2_vl | general |
| [FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B) | 758,833,760 | FastViTHD + Qwen2-0.5B | not stated | DocVQA 82.5, OCRBench 63.9 | Apple research-only, derivatives research-only: **blocks CC BY-NC release** ([licence](https://huggingface.co/apple/FastVLM-0.5B/raw/main/LICENSE)) | fast_vlm | general |
| [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), SmolDocling-256M | 257.5M / 256.5M | SigLIP2-base + Granite-165M | English; JA/AR/ZH experimental | not repeated | Apache-2.0 / CDLA | idefics3 | DocTags (current ResuVL-S) |
| SmolVLM2-500M / 256M | 507M / 256M | SigLIP + SmolLM2 | general | OCRBench 609 (LFM card) | Apache-2.0 | smolvlm | general (current ResuVL-M family) |
| [Dolphin-1.5](https://huggingface.co/ByteDance/Dolphin-1.5) | 397.7M | Swin + 10-layer mBART | zh, en only | n/a | MIT | VisionEncoderDecoder | layout then element parsing |
| [GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) | 560.5M | ViTDet-B ~80M, 1024x1024 + Qwen-0.5B | English and Chinese ([paper](https://arxiv.org/abs/2409.01704)) | n/a | Apache-2.0 | got_ocr2 | plain / formatted text |
| [Florence-2-base / large](https://huggingface.co/microsoft/Florence-2-large) | 231.6M / 776.7M | DaViT + BART-style enc-dec | English-centric | n/a | MIT | florence2 | fixed small input; poor fit for dense pages (my assessment) |
| Donut-base | ~0.2B (no safetensors) | Swin-B + mBART | trained on en/zh/ja/ko ([paper](https://arxiv.org/abs/2111.15664)) | n/a | MIT | donut | JSON by design; input 2560x1920, decoder max 1536 tokens |
| Nougat-base / Pix2Struct-base | 348.7M / 282.3M | | English | n/a | CC BY-NC / Apache | yes | academic papers / screenshots |
| T5Gemma-2 270M-270M | 786M | | Gemma licence, gated | no OCR evaluation | gemma | t5gemma2 | general |

Lower priority: Hiro-MOSS-OCR-0.3B, hayai-ocr (157M), baberu-ocr; I did not review their French evidence.

## Near-misses (>=1B total): pruning or distillation sources
| Model | Total | Notes |
|---|---|---|
| [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B), LightOnOCR-3-1B | 1,005,647,872 | 5.6M over. Pixtral-style ViT 403M + Qwen3-0.6B. French-emphasised, olmOCR 83.2±0.9. The paper's vocabulary pruning (EN/FR) of the v1 model gave 883M (16k vocab) with olmOCR 75.4 vs 76.1 ([paper](https://arxiv.org/abs/2601.14251)). Released pruned versions: [0.9B-16k](https://huggingface.co/lightonai/LightOnOCR-0.9B-16k-1025) (883.6M) and 32k (917M), both from v1 |
| [InternVL3.5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF) | 1,060,897,792 | Qwen3-0.6B + InternViT-300M. lm_head and embeddings are NOT tied (I compared the first 20 MB; they differ), so the count is real. Pruning the vocabulary to 32k gives about 0.82B |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 1,325,258,240 | The "0.9B" claim excludes embeddings. Without the MTP layer it is ~1.1B. French listed, MIT, JSON extraction. MDPBench 67.3 |
| MinerU2.5-Pro-1.2B | 1.156B | MDPBench 71.0 |
| MiniCPM-V-4.6 | 1.300B | |
| Ovis2-1B | 1.267B | |
| TeleOCR | 1.415B | |
| Kosmos-2.5 | 1.375B | |
| LFM2-VL-1.6B | 1.585B | |
| HunyuanOCR (now 1.5) | 2.2B | Licence excludes the EU, so out anyway |
| Teachers/sources | | LightOnOCR-3-4B (4.5B), Qwen3.5-2B/4B, dots.mocr (3B) |

Out of scope or too big, checked: DeepSeek-OCR(-2) 3.3-3.4B, dots.ocr 3.0B, olmOCR 8.3B, Nanonets-OCR 3.8B, Chandra 5.3B, Gemma-4-E2B 5.1B, PaliGemma 2.9B, Aya Vision 8.6B, Qwen3-VL-2B 2.1B, Granite vision 3-4B, Ministral-3-3B 3.8B, MiniCPM-V 4.x >=1.3B, moondream2 1.9B, Qwen2-VL-2B 2.2B. No official moondream 0.5B repo was found.

## Ranked shortlist

1. **LightOnOCR-3-0.8B** (853M). It has the best French evidence (a French benchmark, 70.5; French-heavy training data) and the largest non-embedding LM in the group. It was trained with RL against loops and omissions, which targets our invalid-JSON and left-out-value errors. Downside: the vision encoder is only 100M, French numbers are vendor-run, and it is 9 days old. Effort: new family `Qwen3_5ForConditionalGeneration`, transformers >=5.5.4 in a separate venv. The GatedDeltaNet fast path needs the `fla` and `causal-conv1d` packages, or it falls back to slow PyTorch ops ([docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md)). Both installs need approval.
2. **Xiaomi-OCR-0** (853M without MTP). Same code path as #1, so it is nearly free once #1 works. It is the only candidate trained on prompted schema-to-JSON extraction, which fits our output. Downside: French is not stated, its extraction benchmark is ID cards and flights, and 59 % there is modest.
3. **OvisOCR2** (853M). Same code path. It is the strongest true end-to-end page reader (no layout detector), at 96.58 on OmniDocBench v1.6. Downside: English/Chinese-centred evidence and no French claim. Running #1-#3 plus plain Qwen3.5-0.8B-Base as a control separates "OCR post-training" from "architecture".
4. **PaddleOCR-VL-1.6** (959M). It has the largest and most multilingual vision encoder (466M, French explicit, 109 languages), which targets French reading and omissions. Downside: ERNIE-0.3B is a smaller LM than our current 1B, so field assignment may suffer, and it was trained on crops (page reading must be learned from our data). Effort: new family `paddleocr_vl` (transformers >=5); no maintained HF full fine-tune script known, only the official PaddleFormers/ERNIEKit.
5. **LightOnOCR-2-1B with vocabulary pruned to 32k or 16k** (about 0.87-0.92B after pruning). Strongest French-trained Pixtral-ViT model. Downside: pruning is our own work, validated only on v1 (olmOCR 76.1 to 75.4 at 16k), and the model is superseded by v3. Effort: `lighton_ocr` family plus a pruning script. Take it only if #1 shows the larger ViT matters.

Not recommended: MonkeyOCRv2-B-Parsing (best MDPBench French evidence at 78.4 and Apache data, but remote code and element-level), Surya-2 and FastVLM (licence), Nemotron-Parse (heavy ViT-H, no French evidence), Falcon-OCR (promising at 270M, French undocumented, remote code), the old Florence/Donut/GOT/Dolphin class (low fixed resolution or en/zh only).

Note on MonkeyDocv2: an Apache-2.0 multilingual (French included) document-image corpus of 113M samples may help pretraining. I did not check how much is French.

## Could not verify
- OvisOCR2 and Xiaomi-OCR-0 French support (not stated in cards or reports).
- Falcon-OCR language coverage.
- LightOnOCR-3-0.8B language list beyond "20+ languages"; the HF card lists only `en` while the blog mentions French; only the blog's French benchmark is evidence.
- All benchmark numbers are vendor- or competitor-reported. The MDPBench table comes from the MonkeyOCRv2 authors, who say MDPBench is from the same research line. The fr-bench-pdf2md table is LightOn's own.
- PaddleOCR-VL v1.0 OmniDocBench v1.5 overall (92.86 in the arXiv HTML, a different value in the older card).
- InternVL3-1B per-benchmark OCR numbers (in image-only tables).
- Whether the Surya-2 modified OpenRAIL-M licence permits release under CC BY-NC.
- Qwen3.5 hybrid-attention CPU speed on a laptop.
- Donut exact parameter count (no safetensors).
- Hiro-MOSS, hayai-ocr and baberu-ocr internals.
