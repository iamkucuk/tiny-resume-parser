# Literature review: large-gain methods for sub-1B CV-page-to-JSON (2026-10-14)

Scope: InternVL3-1B-hf track (best 0.728 gold / 0.732 val lenient) and SmolVLM-500M track (best 0.545 / 0.589).
Every number below comes from a source I opened (arXiv HTML/PDF text, or Hugging Face card/config). Sources I could only
see as a search snippet, or only as an abstract, are marked. "(inferred)" = my reasoning, not read.

## 0. Key facts that shape the ranking

1. **SmolVLM-500M already uses the most aggressive compression.** Its config has `scale_factor` 4 (pixel shuffle r=4),
   `image_size` 512, `patch_size` 16, `image_seq_len` 64, `size.longest_edge` 2048, text `max_position_embeddings` 8192
   (HF `config.json`, `processor_config.json`). The SmolVLM paper (2504.05299, sec. 2.2) says InternVL and Idefics3 use r=2
   to protect OCR, but its own Figure 3 (right) shows small VLMs do better with r=4 on its aggregate metric (CIDEr + VQA
   accuracy; no OCR-only number in the text). It also says 135M/360M LMs "struggled beyond 8k" context.
2. **The same 64-token-per-512-px budget can read pages when trained for it.** SmolDocling (2503.11576) is a 256M model on the
   SmolVLM-256M base (SigLIP-B/16 512 + SmolLM2-135M), "aggressive pixel shuffle ... 64 visual tokens per 512x512",
   pages at 144 DPI. After doc-specific training it scores edit distance 0.48 / F1 0.80 on DocLayNet full-page OCR, better
   than GOT (0.61 / 0.69), Nougat (0.62 / 0.66) and Qwen2.5-VL-7B (0.56 / 0.72). So SmolVLM's 16% word reading is mainly missing
   reading training, not a hard token-budget limit (inferred from SmolDocling's result; no ablation in that paper).
3. **Teacher self-agreement 0.912 is the label ceiling**, so InternVL's 0.73 leaves a gap that is mostly losses of the kind you measured:
   parse failures (308 values), omissions (527), wrong field (202). These are generation-structure problems, not reading problems.
   For SmolVLM the gap to InternVL (0.18) is mostly reading (inferred).

## 1. Ranked hypotheses

Effort = hours to implement in the HF transformers script. Compute = GPU-hours (H100) for the first decisive experiment (my estimates, inferred
unless stated). Track: IV = InternVL3-1B, SV = SmolVLM-500M.

| # | Hypothesis (plain words) | Track | Mechanism | Evidence (paper: number) | Failure mode targeted | Effort h | GPU-h | Risk | Owner approval |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Add a "read the page" stage before JSON training for SmolVLM: train page -> plain transcript, using transcripts made by granite-docling-258M on our real pages (no human labels), then the normal JSON fine-tune | SV | Gives the encoder/LM reading skill the base model lacks (16% of words) | Pix2Struct 2210.03347: DocVQA 67.8 full vs 12.2 without screenshot-parsing pretraining; no reading warm-up 56.2. Donut 2111.15664: reading is "the most effective" pretraining task (figure only, no number in text). DocOwl1.5 2403.12895: adding parsing stage 72.8 -> 77.7 DocVQA. SmolDocling: see sec. 0 | SmolVLM misreads/omits because it cannot read small text | 8-12 | 10-20 | Medium: pseudo-transcripts have errors (esp. Turkish/French diacritics, inferred); gain may saturate if LM size (360M) is the limit | granite-docling is already tested on the Mac; running it on TRUBA needs a model download approval (Apache-2.0, 258M) |
| 2 | Generate the JSON one section at a time (basics, work, education, skills, rest), each with its own prompt, then merge | IV, SV | Shorter outputs: a loop or syntax error loses one section, not the page; less omission pressure per output | Alibaba resume parser 2510.09722: Qwen3-0.6B-SFT, three parallel sub-tasks, F1 0.964 on real resumes (Claude-4 in the same pipeline 0.959); ablation removing layout/post-processing drops long-text accuracy 0.858 -> 0.758 (not a split-vs-no-split ablation). ExtractBench 2602.12247: "output volume, not input length or nesting depth, is the dominant predictor of failure" (frontier LLMs, whole-document outputs) | 308 values lost to unparsable pages; loops; omissions | 6-10 | 1 baseline-equivalent training run + 6x decode | Medium: values under another field (18% of teacher disagreement) may get duplicated or dropped across sections; no controlled split/no-split test exists in the literature I found | None |
| 3 | Start the SV-style track from a model that already reads, instead of SmolVLM-500M: first granite-docling-258M (same Idefics3 code path), then LightOnOCR-0.9B-32k | new track / SV | Reading skill comes from the init, fine-tune only teaches the schema | Granite-docling: SmolDocling numbers above (F1 0.80 page text). LightOnOCR-2-1B 2601.14251: OlmOCR-Bench 83.2; trained on 17M->43M pages. No paper shows OCR specialists fine-tune better on KIE; this is inferred | Reading gap on SV | granite 4-6; LightOn 8-16 (custom code, transformers training support "coming soon" on its card) | 6-12 | High for granite (LM only 165M, smaller than SmolVLM's 360M: SmolVLM paper OCRBench 52.6 -> 61.0 from 256M -> 500M LM); medium for LightOn | Yes: new model downloads; granite 258M and LightOn-0.9B (HF lists 0.9B BF16) are <1B. PaddleOCR-VL card lists 1.0B params: not strictly <1B, do not use. Owner should also confirm a third track is allowed |
| 4 | Only for pages where decoding fails (unparsable JSON or token limit hit), re-decode with `no_repeat_ngram_size=4` and/or sampling, instead of using a global repetition penalty | IV, SV | Loops are page-local; a targeted retry does not hurt normal repeated values | Arabic OCR study 2608.22366: loops "associated with greedy decoding"; rerunning failed cases with no_repeat_ngram_size=4 fixed 51/56 (Qwen2.5-VL-7B) and 50/55 (Fanar-2) | 308 lost values on 3 unparsable pages (IV); 12-20% loops (SV) | 3-5 | ~2 (re-score dev) | Low. Different from your failed global repetition penalty. Retry can still produce a bad JSON | None |
| 5 | Render synthetic CV pages with known JSON (new templates, fonts, languages, columns), use them for reading and for JSON training mixed with real pages | IV, SV | Exact labels, unlimited layout/language variety; teaches complete extraction with no teacher omissions | DocDjinn 2602.21824 (KIE, BERT/LiLT/LayoutLMv3): real 76.41 avg F1, synthetic-only 52.97, mixed 76.80. Donut: synthetic data "enough for the document IE task" (SynthDoG, 0.5M per language). Alibaba resume: 2,500 synthetic of 15,500 training resumes | Omissions; SV reading; split/merge noise from teacher | 20-30 (renderer; check what HTML/PDF renderer exists on TRUBA, no installs) | 20-60 | Medium: synthetic-only reaches only ~70% of real in DocDjinn; template realism; mixed gain there is only +0.39 F1 | Maybe: any new renderer package; fake-content generation via GLM is API use only |
| 6 | Combine the 10 existing seed models by field-level voting at output time (keep a value if >= k of 10 models wrote it), instead of or on top of the weight average | IV | Omissions and misreadings differ between seeds; voting recovers items found by several | LMDX 2309.10952: 16 sampled completions + majority vote 54.35 micro-F1 vs 52.85 with one sample (+1.50); ignoring missing entities cost 6.77 | 527 omitted values; 139 misreadings | 4-6 | ~10 (10 models x dev inference) | Low risk, modest gain likely (inferred +0.01-0.03). 10x inference cost; only practical as a teacher for self-distillation on unlabelled pages | None |
| 7 | While training, randomly corrupt some decoder input tokens (targets unchanged) so the model learns to recover after its own mistakes | IV, SV | Reduces the "one error -> loop / collapse" chain | Nougat 2308.13418: perturbation caused no metric drop and "significant" fewer repetitions; failed conversions on out-of-domain pages fell 32%; baseline repetition 1.5% of pages | Loops (SV 12-20%), invalid escapes/missing commas | 3-5 | 1 run | Low-medium: Nougat's pages are 8M real pages; effect on 2.5k pages unknown | None |
| 8 | Build teacher labels from several teacher runs: pick the "medoid" run (highest mean field-F1 against the other runs), or train on the top-2 runs | IV, SV | Removes random omissions/splits that the student cannot predict | Wang et al. 2407.10456: training on top-40 MBR outputs vs one beam output: +1.43 BLEU, +0.045 COMET22 (en-de, small student). Counter-evidence: consistency-filtered GPT-4 labels for BERT gave F1 0.004 lower (2406.17633). LMDX: voting +1.5 | Split/merge (55% of teacher disagreement), "present in one run only" (16%) | 6-8 | 2 runs; teacher API: 2 extra runs x 2,465 pages at ~80 s/page (your memory note), rate-limited at 8 workers | Medium: MT and classification evidence only; extraction evidence is indirect | None (GLM API already approved) |
| 9 | Give the student OCR text as extra input (image + transcript -> JSON), or make it write transcript first, then JSON | IV, SV | Reading is separated from structuring; LM reads text it handles well | DocVQA, DocVLM 2412.08746: raw OCR words (800 tokens) 85.8 (LLaVA-OV), 84.4 (InternVL2), 89.1 (Qwen2-VL) at a small visual budget vs 66.5 / 56.0 / 84.4 without OCR input; VAREX 2603.15118 (abstract only): layout-preserving text +3-18 pp over image alone | SV reading; IV misreads | 8-12 | 10-20 | Medium-high: needs a second model at inference (total > 1B?), longer sequences; transcript-first doubles output length (more loops, inferred) | Yes: pipeline of two models needs an owner decision on the 1B rule |
| 10 | Reduce compression in SmolVLM: pixel shuffle r=4 -> r=2 (256 tokens per 512 tile) or a horizontal 1x4 merge, new connector trained first, then all | SV | 4x more visual tokens per tile; text lines are horizontal | Idefics3 2408.12637: Idefics2 128 vs 64 tokens "helped OCR tasks after fine-tuning with image splitting". DocOwl1.5: H-Reducer 1x4 DocVQA 72.8 vs 2x2 71.8 vs 2x8 69.2. GOT: 1024x1024 -> 256 tokens (same 64 px/token as SmolVLM) reads dense pages at edit distance 0.035 | Reading | 10-16 (connector shape changes: `scale_factor`, linear in = 768 x r^2; re-init) | 30-50 | High: weak evidence that the connector is the bottleneck (see sec. 0, item 2); sequence length: 13 tiles x 256 = 3,328 image tokens + ~3,000 answer is near the 8,192 limit; 20 tiles would exceed it (inferred arithmetic) | None |
| 11 | Weight loss on value tokens higher than on JSON syntax tokens (or the reverse for syntax boundaries) | IV, SV | Syntax tokens are ~free to predict; uniform averaging dilutes value-token gradient | Only indirect: 2603.23885 (abstract) lists "structure-token optimization" among its recipe and links repetition to unstable structure; TAB-PO 2603.00025 (abstract): +11.59% relative over SFT on ontology-critical metrics, a preference-optimisation method, not SFT weighting. No source I opened gives an SFT value-vs-syntax weighting ablation | Omissions, misreadings | 3-4 | 1 run | Medium; evidence weak | None |

Ordering logic: rows 1-3 target the two largest gaps (SV reading, IV structure losses) with direct or near-direct evidence; row 4 is the cheapest;
row 5 is the only one with a path to unlimited exact labels but costs most to build.

## 2. Hypothesis sections

### H1. Reading stage with pseudo-transcripts (SV)
- **Change:** insert Stage R: input = CV page image (same tiling as the JSON run), target = plain text transcript in reading order, from granite-docling output with DocTags stripped. Train 1-3 epochs on the 2,465 training pages only (exclude gold 1-50 and the 79 validation pages; their images must not enter any stage). Then run the unchanged JSON fine-tune from that checkpoint. Variant: freeze encoder in Stage R, unfreeze in the JSON stage with encoder LR ~100x below LM LR (SmolDocling: 2e-4 vs 2e-6 once unfrozen).
- **Baseline:** current best SV recipe (0.545 / 0.589), same seed count; one change = Stage R.
- **Measure:** (a) zero-shot word-reading rate after Stage R (was 16%), (b) loop rate, (c) lenient F1 on both dev sets.
- **Confirm:** reading rate > 80% and F1 up by more than 0.02 on both sets. **Reject:** reading rises but F1 flat (then the 360M LM or label noise is the limit; go to H3 or H8).
- **Caveat read in sources:** granite-docling card: English supported, Japanese/Arabic/Chinese "experimental"; Turkish and French are not listed (my reading of the card). You measured 96% word reading on your pages, so check Turkish diacritics before trusting transcripts.

### H2. Section-wise generation (IV first, then SV)
- **Change:** split each teacher JSON by top-level key into 4-6 targets (basics / work / education / skills+languages / projects+certs+awards+volunteer+interests+extra). Prompt names the section. Train one model on all pairs (same total target tokens as the baseline). Inference: one decode per section (batch them), then merge. Emit empty list/null for an absent section (LMDX: omitting missing entities cost 6.77 micro-F1, 54.35 -> 47.58).
- **Baseline:** current best single-run InternVL recipe, same data and epochs.
- **Measure:** F1, number of unparsable pages, values lost to parse failure (was 308 on 3 pages), omissions, "under another field".
- **Confirm:** unparsable-page loss drops to under ~100 values and F1 up more than 0.02 on both sets. **Reject:** cross-section errors (duplicates, wrong section) offset the gain.
- **Optional second pass (inferred, no source):** pass 1 lists entity heads (company + title), pass 2 fills each entry.

### H3. Reading-specialist initialisation
- **Change:** granite-docling-258M first (it is Idefics3 and has the same processor family, so the existing SV script should run with a different `--model`, new JSON-schema prompt, DocTags tokens ignored). Run the standard recipe unchanged. If it beats SV-500M's 0.545 / 0.589 beyond spread, try LightOnOCR-0.9B-32k (needs custom model code; vocab pruned to 32k: its card shows -3.0 overall on OlmOCR-Bench vs 151k vocab, with -8.8 on multi-column, so check Turkish/French coverage).
- **Baseline:** SV-500M best.
- **Confirm:** higher F1 and loop rate < SV's 12-20%. **Reject:** granite's LM (165M) caps JSON quality (compare: SmolVLM 256M -> 500M LM raised OCRBench 52.6 -> 61.0).
- **Do not use:** PaddleOCR-VL (HF card: 1.0B params listed); Qwen2-VL-2B (675M ViT + 1.5B LM, over the limit); olmOCR/dots.ocr (not opened, and larger).

### H4. Targeted loop recovery at decode time
- **Change:** after greedy decode, if the JSON does not parse or the token limit was hit, re-run only that page with `no_repeat_ngram_size=4` (and one temperature-0.3 sample as a second try). Keep JSON repair after that. Keep greedy for all other pages.
- **Baseline:** same checkpoint, current decode. No retraining.
- **Measure:** F1 on pages that failed; overall F1; fraction of legitimately repeated values broken (check by diff on pages that did not fail: must be unchanged).
- **Confirm:** >50% of failed pages recovered (Arabic study: 51 of 56) and overall F1 up by more than 0.02. **Reject:** retries yield other unparsable JSON.

### H5. Synthetic rendered CV pages
- **Change:** take teacher JSONs of training pages only (never gold/val), re-render with 10-30 new HTML/CSS (or Pillow) templates: 1-2 columns, sidebars, different fonts/sizes (include small text), languages (EN/FR/TR), dates formats. Optionally ask GLM to write extra fake CVs in the schema (text only). Output: image + exact JSON + exact reading-order transcript. Mix 1:1 with real in the JSON stage; also use the transcripts for H1's Stage R.
- **Baseline:** best recipe on real only. One change = add N synthetic pages (try N = 1,000 and 3,000).
- **Confirm:** F1 up on both dev sets and omission count down. **Reject:** synthetic-only gives < 50% of real-only F1 and mixing adds nothing (DocDjinn pattern: mixed +0.39 over real).
- **Privacy:** rendering real personal content: keep local (project rule on failure_report.html applies).

### H6. Output-level voting over the 10 seeds
- **Change:** decode dev pages with each of the 10 models, normalise values, keep a (path, value) pair if at least k models produce it (try k = 3, 5, 6). Compare with the uniform average (0.728 / 0.732).
- **Confirm:** > 0.02 above the average on both sets, recall rises toward precision (0.76). **Reject:** no gain over the soup. If it works, use it as an offline teacher to label unlabelled CV pages for a student (self-distillation; no source opened for gains, inferred).

### H7. Decoder-input perturbation
- **Change:** in the collator, with probability p per sequence, replace a random 1-10% of the decoder input tokens (not the labels) with random vocabulary tokens. Nougat replaces tokens until a sampled value exceeds a 10% threshold; copy that rule or use a fixed 5%.
- **Baseline:** same recipe without it, 3 seeds each (loop rate varies).
- **Confirm:** loop/unparsable rate down by about a third (Nougat: 32% fewer failed pages out of domain) with F1 not lower. **Reject:** F1 down on gold.

### H8. Multi-run teacher labels
- **Change:** collect 2 extra GLM runs for each training page (3 total). Label A = medoid by mean pairwise field-F1. Label B = top-2 runs both used as separate targets (MBR-n style). Compare A and B with the single-run labels on the same pages and epochs.
- **Confirm:** > 0.02 on both dev sets; check split/merge and omission counts. **Reject:** no gain: the classification study (2406.17633) already found consistent-only labels gave -0.004 F1 for BERT, so a null result is plausible.
- **Cost note:** extra teacher runs for 2,465 pages at ~80 s/page: ~110 worker-hours total, at the 429 limit around 4 workers (from your memory notes).

### H9. OCR text as extra input (hybrid)
- **Change:** prompt = image + granite-docling transcript. Train to JSON. Compare image-only.
- **Confirm:** large gain on SV (> 0.05) at acceptable total size. **Needs owner decision:** a two-model system is 258M + student; also a text-only Qwen3-0.6B on layout-ordered text reached 0.964 F1 in 2510.09722 (their labels, their layout parser), which suggests the text route can work but it is not an image model.

### H10. Less pixel shuffle in SmolVLM
- **Change:** set `scale_factor` 2, re-init the connector (input dim 768x4 = 3,072 instead of 12,288), train connector alone for a short alignment stage (encoder and LM frozen) then full. Cap tiles at 13 (3,328 image tokens).
- **Baseline:** SV best at the same tile count. **Confirm:** reading rate rises beyond what H1 gives; **reject:** no difference after H1 (reading was not budget-limited).

### H11. Token-weighted loss
- **Change:** compute per-token weights from the JSON tokenisation: value tokens 1.0, key/punctuation tokens 0.3, EOS and section-closing brackets 2.0. Compare with uniform.
- **Confirm:** fewer unclosed/loop endings and F1 > spread. Low priority: no direct evidence.

## 3. Answers to the specific questions

**Q1. How strong small doc VLMs get reading ability (sources opened).**
- Donut 2111.15664: pre-train by reading all text in reading order (pseudo-OCR); 11M IIT-CDIP scans (commercial OCR labels) + SynthDoG 0.5M per language x 4; resolution 2560x1920; CORD F1 84.1 / accuracy 90.9 vs LayoutLMv2 78.9 / 82.4 (which needs OCR). Resolution matters: "grows rapidly" with input size; CORD at 1280x960 gave 91.1 accuracy.
- Pix2Struct 2210.03347: 80M screenshots to simplified HTML (50% masked); 30K-step warm-up on rendered text; Table 2 DocVQA 67.8 / 56.2 (no warm-up) / 12.2 (no parsing).
- GOT-OCR2.0 2409.01704: 580M (80M encoder + Qwen-0.5B); stage 1 encoder with OPT-125M on ~5M pairs (3M scene, 2M doc); stage 2 joint training with synthetic engines; stage 3 post-train (8K tokens, 80% data resampled from earlier stage); 1024x1024 -> 256 tokens; English page edit distance 0.035, F1 0.972.
- SmolDocling 2503.11576: three stages (freeze encoder; unfreeze at LR 2e-6 vs 2e-4; fine-tune all); pre-training data DocLayNet-PT 1.4M pages, Docmatix 1.3M; 4 epochs on 64 A100; names "endless repetition" a typical defect, offers no fix.
- Florence-2 2311.06242: 0.23B/0.77B; 384 -> 768 px high-res stage; text labels from Azure OCR API; no ablation isolating OCR.
- Qwen2-VL-2B 2409.12191: 675M ViT + 1.5B LM (too big), 1.4T tokens, DocVQA 90.1.
- PaddleOCR-VL 2510.14528 (1.0B on HF): stage 1 29M pairs, stage 2 2.7M; pseudo-labels from an expert pipeline refined by LLMs plus a hallucination filter; argues element-level recognition avoids long-sequence hallucination (no ablation number).
- LightOnOCR-2-1B 2601.14251: teacher-labelled 17M-43M pages, RLVR cut looping 1.14% -> 0.50%, OlmOCR-Bench 81.8 -> 83.2.
- Not opened (no claims used): olmOCR 2 paper itself (only search summaries), dots.ocr, PaliGemma, Nougat beyond the repetition section.
- **Does a reading stage help extraction, by how much?** Closest evidence: Pix2Struct (DocVQA +55.6 points from parsing pretraining), DocOwl1.5 (+4.9 DocVQA, +3.6 TabFact, +2.5 ChartQA from a parsing stage), Donut (reading is best pretraining task; figure, no text number). None of them is a sub-1B model on dense multi-language CV JSON, so the size of the gain for us is unknown (inferred).

**Q2. SmolVLM connector and resolution.** See sec. 0 item 1 and H10. The paper has no tile-size or OCR-specific pixel-shuffle ablation in text; Figure 3 (right) shows r=2 vs r=4 only on an aggregate metric.

**Q3. Long structured generation.** Evidence is thin and mostly qualitative: ExtractBench (output volume predicts failure; trailing commas and truncated JSON among failure types: 31 and 9 of 103 invalid), Alibaba pipeline (subtask decomposition; their post-processor re-extracts text from source lines, +7.7 long-text accuracy), LMDX (explicit empty values, voting), Nougat (perturbation), Arabic OCR study (targeted rerun). Output serialisation: Dripper 2511.23119: compact format vs JSON F1 0.8726 vs 0.8772 (similar), inference time 3301 s -> 2158 s; the Alibaba paper says YAML/Markdown shorten tokens but JSON "yields the most stable results" (no number). So format changes cut time, not errors.

**Q4. KIE tricks with largest gains.** Reading/parsing pretraining (Pix2Struct, Donut, DocOwl1.5), input resolution (Donut), explicit empty-field output and coordinate/position information (LMDX: line indices instead of coordinates lost 14.98 micro-F1), sample voting (LMDX +1.5), exact-label synthetic data (DocDjinn).

**Q5. Noisy teacher labels.** MBR-n distillation +1.43 BLEU (MT); LMDX voting +1.5; BERT consistency filter -0.004 (classification). Net: moderate hope, no proof for extraction.

**Q6. Other.** Synthetic rendering (H5), OCR hybrid (H9), staged unfreeze with small encoder LR (SmolDocling), checkpoint souping plus RLVR with a loop penalty (LightOnOCR; you already tried GRPO).

## 4. Rejected ideas

- **YAML/Markdown/short-key serialisation as an accuracy lever:** Dripper: same F1, faster; Alibaba: JSON most stable. May cut loops (inferred); not tested as a main hypothesis.
- **Bigger vision encoder for SV:** SmolVLM paper: SigLIP-SO400M with a 360M LM +11.6% on its aggregate metric at +66% parameters; breaks the 1B limit for the 500M model.
- **Qwen2-VL-2B, PaddleOCR-VL (1.0B listed), LightOnOCR-1B (full, 1B), dots.ocr, olmOCR-7B:** at or above the 1B limit.
- **Region crops / element-level parsing (PaddleOCR-VL, Dolphin):** needs layout detection and value-to-region alignment, a second model; Dolphin abstract gives no numbers (only abstract opened). Revisit only if H2 shows section splits help.
- **Another global RL / DPO round:** your GRPO and DPO results were null/negative; LightOn's RLVR gain (+1.4) came with 17M+ pages of pre-training.
- **Confidence-filtered labels alone:** the one opened test (2406.17633) shows no gain.
- **More decoding penalties applied globally:** already lowered F1; Arabic study itself warns they hurt legitimate repeats.

## 5. References (all opened unless marked)

- SmolVLM, arXiv 2504.05299 (2025): https://arxiv.org/abs/2504.05299
- SmolDocling, arXiv 2503.11576 (2025): https://arxiv.org/abs/2503.11576
- GOT-OCR2.0, arXiv 2409.01704 (2024): https://arxiv.org/abs/2409.01704
- Florence-2, arXiv 2311.06242 (2023): https://arxiv.org/abs/2311.06242
- Donut, arXiv 2111.15664 (2021): https://arxiv.org/abs/2111.15664
- Pix2Struct, arXiv 2210.03347 (2022): https://arxiv.org/abs/2210.03347
- DocOwl 1.5, arXiv 2403.12895 (2024): https://arxiv.org/abs/2403.12895
- Idefics3 / "Building and better understanding VLMs", arXiv 2408.12637 (2024): https://arxiv.org/abs/2408.12637
- Qwen2-VL, arXiv 2409.12191 (2024): https://arxiv.org/abs/2409.12191
- PaddleOCR-VL, arXiv 2510.14528 (2025): https://arxiv.org/abs/2510.14528 ; HF card https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- LightOnOCR-2, arXiv 2601.14251 (2026): https://arxiv.org/abs/2601.14251 ; cards https://huggingface.co/lightonai/LightOnOCR-1B-1025 , https://huggingface.co/lightonai/LightOnOCR-0.9B-32k-1025
- Towards Real-World Document Parsing (abstract only), arXiv 2603.23885 (2026): https://arxiv.org/abs/2603.23885
- Nougat, arXiv 2308.13418 (2023): https://arxiv.org/abs/2308.13418
- Arabic manuscript OCR study, arXiv 2608.22366 (2026): https://arxiv.org/abs/2608.22366
- Layout-Aware Parsing Meets Efficient LLMs (resume), arXiv 2510.09722 (2025): https://arxiv.org/abs/2510.09722
- LMDX, arXiv 2309.10952 (2023): https://arxiv.org/abs/2309.10952
- ExtractBench, arXiv 2602.12247 (2026): https://arxiv.org/abs/2602.12247
- DocVLM, arXiv 2412.08746 (CVPR 2025): https://arxiv.org/abs/2412.08746
- Dripper, arXiv 2511.23119 (2025): https://arxiv.org/abs/2511.23119
- DocDjinn, arXiv 2602.21824 (2026): https://arxiv.org/abs/2602.21824
- VAREX (abstract only), arXiv 2603.15118 (2026): https://arxiv.org/abs/2603.15118
- SAYRE (abstract only), arXiv 2607.04636 (2026): https://arxiv.org/abs/2607.04636
- Don't Throw Away Data: Better Sequence KD, arXiv 2407.10456 (2024): https://arxiv.org/abs/2407.10456
- Better Instruction-Following Through MBR, arXiv 2410.02902 (2024): https://arxiv.org/abs/2410.02902
- Knowledge Distillation in Automated Annotation, arXiv 2406.17633 (2024): https://arxiv.org/abs/2406.17633
- TAB-PO (abstract only), arXiv 2603.00025 (2026): https://arxiv.org/abs/2603.00025
- Dolphin (abstract only), arXiv 2505.14059 (2025): https://arxiv.org/abs/2505.14059
- Model soups (abstract only), arXiv 2203.05482 (2022): https://arxiv.org/abs/2203.05482
- HF cards/configs: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct (config.json, processor_config.json), https://huggingface.co/ibm-granite/granite-docling-258M
- Not opened, not used: olmOCR 2 (arXiv 2510.19817), NEXT-EVAL, Dessurt details (only abstract opened).
