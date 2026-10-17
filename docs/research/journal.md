# Research journal — CV page image → ATS JSON, autonomous loop on TRUBA

Started 2026-10-13. Goal set by the user: field F1 reasonably above 0.9 on held-out reviewed gold pages.
Differences that are not errors (case, spacing, punctuation such as "Nvidia" vs "NVIDIA") may stay; any
difference in letters or digits ("09.25" vs "09.20", "Tunisie" vs "Tunasie") is an error.
Each entry: hypothesis → why → what was run → what came out → what went right / wrong → next hypothesis.
Short per-run summaries also go to `DEVLOG.md`; full result tables live here.

## Tracks (kept current; read this first) — updated 2026-10-14 07:40
Two student models are tracked separately. Scores of one are never compared with, or averaged into, the other.
| track | base model | run-name prefix | best so far (lenient, gold / val) | reference to beat |
|---|---|---|---|---|
| InternVL (main) | OpenGVLab/InternVL3-1B-hf, 0.94 B | Z*, ZH*, ZK*, S_soup* | S_soup10_vision 0.728 / 0.732 (val-chosen average) | default recipe + extra pages (ZK), 6-seed mean 0.716 / 0.721 (gain below 0.02) (old recipe 9-seed mean 0.705 / 0.708) |
| SmolVLM (second) | HuggingFaceTB/SmolVLM-500M-Instruct, 0.51 B | SV_* (averages: SV_soup*) | SV_soup3_tiles30 (average of 3 30-tile seeds) 0.644 / 0.676; 3-seed mean 0.619 / 0.654 | 30 tiles beat 12 by +0.043 / +0.032 on 2-seed means; 20 tiles = 30; 48 no gain; key-by-key run scoring |
| granite-docling (third) | ibm-granite/granite-docling-258M, 0.26 B | GD_* | GD_soup4_lr1e-4 (average of 4 lr 1e-4 seeds) 0.660 / 0.649; single-seed mean 0.616 / 0.616 | 4-seed average +0.044 / +0.033 over the seed mean; 30/48 tiles lower; key-by-key run queued |
- Safeguards in code: evaluate.py picks the processor from the checkpoint's config (internvl / idefics3); soup.py stops
  if tensor names differ, so a mixed average cannot be built; SmolVLM accepts only --max-patches 12.

## Protocol (fixed before any new run)
- **Dev set (used for every decision):** gold pages 1–50 (`labels_gold_v6_verified.jsonl`, reviewed, CVs never trained
  on; same 50 pages as every earlier score) + the 79-page validation split (5% of training CVs by hash, GLM labels,
  not reviewed).
- **Test set (used only at milestones):** gold pages 51–254 (204 pages), `evaluate.py --set gold --start 50`.
  Keeping it untouched means the final number is not inflated by picking the best run on the dev pages.
- **Metrics** (`src/rescore.py`): *strict* F1 = exact (path, value) match (`evaluate.pairs`), pooled over pages;
  *lenient* F1 = the same after casefolding and removing every character that is not a letter or digit. Accents
  are kept (é ≠ e counts as an error). Lenient F1 is the number the 0.9 goal refers to; strict is reported too.
- **Noise:** paired bootstrap over pages (2,000 resamples) and, from wave 1, a repeat run with another seed.
- **Compute:** TRUBA `kolyoz-cuda` (H100 80 GB / H200), project copy `/arf/scratch/fkucuk/cv_parser_test/cv_parser`,
  job script `truba/exp.slurm` (train → score every epoch on the dev pages → rescore).

## Starting point (before this loop)
| Model | Dev gold-50 strict F1 | lenient F1 | Training pages (120) strict / lenient |
|---|---|---|---|
| lora_v4 epoch 6, free decoding | 0.639 | 0.656 | — |
| lora_v6 epoch 8 (12 tiles, rank 32), free | 0.630 | 0.650 | 0.853 / 0.862 |
| lora_v6 epoch 8, constrained decoding | 0.641 | 0.662 | — |

Lenient scoring adds only ~0.02, so ~0.25–0.3 of real errors separate us from 0.9.
Per section (v6 constrained, strict): work 0.69, skills 0.68, projects 0.58, education 0.58, basics 0.73,
languages 0.80, certificates 0.53, interests 0.43, awards 0.43, extra 0.

## Observation 0 — the model does not reproduce even its training pages (2026-10-13)
lora_v6 (train loss 0.036 in the last epoch) scores only 0.85 F1 on 120 pages it was trained on 8 times.
Lenient errors on those pages: 809 values missing, 376 misread, 216 under the wrong key, 1,418 extra values.
One page inspected value by value (`1629932620090`):
- *Splitting choices:* the label has `education.honors: "avec félicitation de jury"` as its own value; the model
  appends it to `education.summary`. The label puts "Participé à projet: Centrale photovoltaïque…" in
  `work.highlights`; the model puts it in `work.summary`. The label has `area` "Physique et Chimie" separately;
  the model leaves it inside `studyType`.
- *Real misreads:* `ETELEC` → `TELECI`, `MOSTAYYSSER` → `MOSTAYSSER`, a Cyrillic "р" inside "aéраulique".
- *Typography:* "Stagiaire -Stage" (label) vs "Stagiaire –Stage" (model): lenient scoring forgives this.

**Hypothesis H0 (label ceiling):** many teacher decisions (which field, where to split) are not determined by the
page, so GLM itself would not repeat them exactly. If GLM-vs-GLM F1 is far below 1, then exact field F1 > 0.9 is
not reachable by any student, and the conventions must be made deterministic (by code or a stricter teacher prompt).
**Test:** relabel the 41 CVs behind the 50 dev gold pages with the same teacher and settings
(`runs/label_ceiling/glm_rerun.jsonl`), score run 2 against run 1 (`labels_gold_v6.jsonl`) and against the
reviewed gold.

**Result H0 (2026-10-13 02:40):** all 51 pages relabelled (51 ok, 33 min, 4 workers).
| Comparison on the 50 dev gold pages | strict F1 | lenient F1 |
|---|---|---|
| GLM run 2 vs GLM run 1 (same teacher, same prompt) | 0.899 | 0.912 |
| GLM run 2 vs reviewed gold | 0.899 | 0.912 |
| GLM run 1 vs reviewed gold (gold = run 1 + 96 review fixes on 254 pages) | 0.996 | 0.997 |
Per section, run 2 vs gold (lenient): work 0.94, skills 0.91, projects 0.91, education 0.89, basics 0.94,
languages 0.96, volunteer 0.96, certificates 0.88, interests 0.73, awards 0.70, extra 0.32.
**Interpretation:** the teacher repeats its own field decisions only ~91% of the time. The gold labels are one
draw of those decisions. A student that learns the teacher's *most likely* decision can score somewhat above a
second random draw (it is not adding its own sampling noise), but values that are a coin flip for the teacher
(extra vs skills, award vs certificate, how an education line is split) are a coin flip for the student too.
So ~0.9 lenient is near the ceiling of the current labels; clearly above 0.9 needs labels whose conventions are
deterministic. **Next hypothesis H5:** making the training labels self-consistent (fixed rules applied by code, and
where code cannot decide, majority vote over several teacher runs) raises dev F1 beyond what model size does.
Caution: the gold labels carry run-1 conventions; any convention rule must be applied identically to gold
(so it is a fixed canonical form of the same information, not a change of what is scored) and be listed here.

**Where the two GLM runs disagree** (lenient, 50 dev pages, 4,612 gold values, ~400 disagreeing):
220 the same text split or merged differently (a skill line kept whole vs split; two highlight lines merged;
"TAV Technologies (TAV Information Services Co.)" as one name vs two), 74 the same value under another key
(interests.name ↔ interests.keywords 21, skills.name ↔ skills.keywords 13, projects.name → description 5),
66 present in one run only, 41 different text (one run reads "taillwind", the other "tailwind").
**Can code remove the ambiguity?** Canonical forms applied to both runs (scratchpad `canon_test.py`), lenient F1
run 2 vs gold: none 0.9125; split keyword-type lists on `, ; | • ·` 0.918; also drop "Label:" prefixes 0.920;
merge interests.name+keywords 0.917; skill groups without keywords → keywords 0.915; all three 0.927.
**Conclusion:** code rules gain only ~0.015; most disagreement is judgement no simple rule fixes. The labels'
ceiling stays ~0.92 lenient. The student (0.65) is far below it, so the student is the bottleneck now; label
work (H5) is parked until a model reaches ~0.85.

## Wave 1 (submitted 2026-10-13 ~01:35): capacity and variance
All use the lora_v6 input (12 tiles) and are scored at every epoch on the dev pages.
| Run | Hypothesis | Setup | GPUs |
|---|---|---|---|
| A_v6_seed1 | Run-to-run noise is small enough to trust single-run differences of ~0.02 | lora_v6 recipe, seed 1 | 1 H100 |
| B_fullft_1b | LoRA r32 limits fitting (training pages only 0.85); training all LM + projector weights fits better and generalises better | full fine-tune 1B, lr 1e-5, 6 epochs | 1 H100 |
| C_2b_lora | A 2B model reads and structures better | InternVL3-2B, lora_v6 recipe | 1 H100 |
| D_8b_lora | Upper bound from size: if 8B is not near 0.9, size is not the main limit | InternVL3-8B, LoRA r32 + projector, 6 epochs, effective batch 16 | 4 H100 (DDP) |

Also running: `cvp_dpo_gen` — lora_v6 answers on all 1,393 training pages, the data for preference training.

### Wave 1 results so far (2026-10-13, dev pages; strict / lenient F1)
| Run | Best epoch so far | Gold-50 | Validation (79) | Notes |
|---|---|---|---|---|
| lora_v6 (reference, 1B LoRA, seed 42) | ep8 constrained | 0.641 / 0.662 | — | |
| A_v6_seed1 (same recipe, seed 1) | ep5 | 0.643 / 0.661 | 0.634 / 0.650 | still training to ep8 |
| B_fullft_1b (1B full fine-tune) | ep5 | 0.675 / 0.693 | 0.654 / 0.669 | ep6 0.689 gold; done, 21 GB of checkpoints |
| C_2b_lora (2B LoRA) | ep4 | 0.714 / 0.735 | 0.684 / 0.703 | later epochs pending |
| **D_8b_lora (8B LoRA, 2 GPUs)** | **ep2** | **0.768 / 0.791** | **0.751 / 0.766** | training 41 min for 6 epochs; scoring per epoch is the slow part |

**What went right:** model size is by far the strongest lever: 1B 0.66 → 2B 0.735 → 8B 0.79 lenient, and the
validation split agrees. Full fine-tuning of 1B adds ~+0.03 over LoRA (smaller than one size step).
Per section (lenient), 8B vs lora_v6: education 0.59 → 0.80, projects 0.61 → 0.80, certificates 0.53 → 0.71,
skills 0.70 → 0.80, work 0.71 → 0.80. `extra` stays ~0 (22 values; the teacher itself is inconsistent there, H0).
**Error types** (scratchpad `errcat.py`, unmatched gold values on the 50 pages, gold total ~4,600):
| | lora_v6 | 2B | 8B ep2 |
|---|---|---|---|
| same text split/merged differently | 554 | 317 | 275 |
| missing | 476 | 245 | 213 |
| misread (same field, ≥0.8 similar text) | 279 | 229 | 258 |
| right value under another key | 260 | 166 | 126 |
| predicted values with no gold match | 1,533 | 1,210 | 988 |
Size reduced structure errors (split/merge, wrong key, missing) by ~half, but *misreads did not drop*.
A sample of 40 8B misreads: real character errors mostly in names and rare words ("ANTABLI"→"ANTABI",
"BOUSRIH"→"BOUSRIRH", "Siliana"→"Silliana", "BALTI"→"BALT", a date "2018-08"→"2018-02"), plus label quirks
(teacher wrote "angulaire.js"; "Agil" on the page vs "Agile") and spacing that lenient scoring already forgives.

## Wave 2 (submitted 2026-10-13): resolution and stronger readers
- **H6 — resolution (F_8b_24tiles):** misreads did not fall with size, so they are limited by the image the model sees
  (12 tiles of 448 px; a dense A4 page is downscaled). 24 tiles should fix name/date misreads. One change vs D.
- **H7 — larger model (G_14b_lora):** the size trend has not flattened (2B→8B +0.056); InternVL3-14B (30 GB).
- **H8 — newer model of the same size (H_35_8b_lora):** InternVL3.5-8B (same transformers class, Qwen3 language
  model, reported better OCR/document scores). One change vs D.

**Correction (2026-10-13, user):** only models under 1B parameters are allowed (InternVL3-1B now, SmolVLM-500M
later). The 2B and 8B runs (C, D, F) and the planned 14B / InternVL3.5-8B runs (H7, H8) were outside the task. They
are not candidates; the 14B and 3.5-8B downloads were deleted. Their results are kept above only as a measurement:
size reduced structure errors but not misreads.
**1B results that are in scope:** A (seed 1) ep8 0.662 lenient vs lora_v6 ep8 0.650 free → run-to-run spread ~0.01–0.02.
B (full fine-tune) ep5 0.693 → best 1B so far. E (preference training from lora_v6) *lowered* F1 every step:
0.583 at step 30, 0.556 at 60, 0.539 at the end (lora_v6 0.650) → negative result.

## Wave 2 — 1B only (submitted 2026-10-13 ~10:45; started ~11:00, then ~45 min lost to a Lustre hang before Python ran)
Base recipe = B (full fine-tune InternVL3-1B, lr 1e-5, 6 epochs, 12 tiles), best 1B so far (0.693 lenient). One change each:
| Run | Change | Hypothesis |
|---|---|---|
| G_ft_24tiles | 24 tiles | misreads (names, dates) come from image downscaling; more tiles fix them |
| H_ft_lr5e6_8ep | lr 5e-6, 8 epochs | B peaked at ep5 then fell; a slower schedule overfits less |
| I_ft_seed1 | seed 1 | run-to-run spread of full fine-tuning (needed to trust ±0.02 differences) |
| J_ft_lr2e5 | lr 2e-5 | B underfits (training-page F1 of LoRA was only 0.85); a higher lr fits better |
| K_ft_canon | training labels in canonical form (`src/canon.py`) | the student wastes capacity on the teacher's inconsistent split/merge choices (largest error type) |

**H9 check before K (no training):** applying `src/canon.py` (split list fields on `, ; | • ·`; interests one entry per
interest; skill groups without keywords → keywords; no page text dropped) to both answers and gold:
B ep5 gold 0.693 → 0.713 lenient, validation 0.669 → 0.684; lora_v6 constrained 0.662 → 0.685. The canonical
training file changes 926 of 1,472 pages and passes the schema on all. From now on every run is reported both
raw and canonical (`rescore.py --canon`, written to `rescore_canon.json`); the 0.9 goal is judged on the canonical
lenient score only if the rules stay "no text dropped" — list the rule set with the number.
| L_ft_vision (queued 1601853) | vision encoder trained too (`--train-vision`, new flag) | misreads did not fall with model size, so they may come from the frozen encoder, not the language model |
| M_ft_combo (queued) | canonical labels + 24 tiles + vision encoder trained, together | not a one-change test: the earliest strong 1B candidate if G, K and L each help; compared against them afterwards |
| N_ft_canon_seed1 (queued) | K with seed 1 | run-to-run spread of the canonical-label recipe, so a K gain can be told apart from noise |

### Wave 2 (1B) — epoch 3 scores, gold-50 lenient F1 (raw / canonical via `rescore.py --canon`)
| Run | raw | canonical |
|---|---|---|
| B full FT, seed 42 | 0.668 | 0.683 |
| I full FT, seed 1 | 0.660 | 0.684 |
| J lr 2e-5 | 0.638 | 0.660 |
| K canonical training labels | 0.675 | **0.702** |
Early read (one epoch): K leads by ~+0.018 canonical over both seeds of B (seed gap 0.001 canonical, 0.008 raw);
J is behind and low on recall (0.61) — higher lr fits the training pages harder (train loss 0.012) but leaves out values.
Validation split (79 pages), epoch 3, lenient canonical: B 0.659, I 0.651, **K 0.643**. So K leads on gold-50
(+0.018) but trails on validation (−0.008 to −0.016): at epoch 3 the canonical-label effect is within noise; no
conclusion until best epochs and the K repeat (N, seed 1) are in. I epoch 4 gold raw 0.686 (B ep4 0.683).
Epoch 4 (gold-50 raw lenient): I 0.686, K 0.683, J 0.661 (B ep4 0.683). Validation ep4: I 0.660.
H (lr 5e-6, 8 epochs) ep3 gold 0.614 — slower lr is behind at the same epoch, as expected; judged at its best epoch.

### H10 — data size is the main limit for 1B (learning curve first)
Every 1B variant lands at 0.68–0.69 lenient (lr, seed, canonical labels move ±0.02); training pages score 0.85, so the
gap to unseen pages (~0.16) is large. **Test (O_ft_half, job 1601917):** the B recipe on the first 696 of 1,393 training
pages (same 6 epochs, so half the steps). Reading: if half data is clearly worse (≥0.03 below B on both dev sets),
more labelled pages should help and labelling more is the next step; if equal, data size is not the limit.
**Where more pages would come from:** (a) ~557 Roboflow CVs (~670 pages) left out because a page is ≤640 px tall
(low resolution, `cv_ids_full_size.json`); (b) `hadikp/resume-data-pdf`, 8,905 single-page English resume images in
`data/raw/` (different template style from the mostly French/Tunisian Roboflow CVs). Both need GLM teacher labels
(measured: 51 pages in 33 min with 4 workers ≈ 1.5 pages/min, so 1,000 pages ≈ 11 h).
12:11 — G (24 tiles) ep3 gold 0.637 (B ep3 0.668); K (canonical labels) ep5 gold 0.672 (B ep5 0.693, I ep5 0.686);
N (canonical, seed 1) ep3 gold 0.648. So far neither 24 tiles nor canonical training labels beat the baseline on raw score.

### Wave 2 (1B) — best-epoch results so far (lenient F1, raw / canonical)
| Run | Gold-50 best | Validation best |
|---|---|---|
| B full FT lr 1e-5 seed 42 | ep5 0.693 / 0.713 | ep5 0.669 / 0.684 |
| I same, seed 1 | ep6 0.688 / 0.707 | ep5 0.677 / 0.687 |
| J lr 2e-5 | ep5–6 0.693 | ep6 0.688 |
| K canonical training labels | ep4 0.683 / ep3,6 0.702 | ep6 0.667 / 0.678 |
| G 24 tiles | ep4 0.641 (so far) | ep3 0.640 |
| H lr 5e-6, 8 ep | ep5 0.653 (so far) | ep4 0.634 |
**Conclusions:** seed spread at best epoch < 0.01 on both sets. Canonical training labels (H9) do not help, even
scored canonically (−0.005 to −0.01) → rejected; canonical *scoring* adds ~+0.015–0.02 to every run (metric only).
lr 2e-5 reaches the same best as 1e-5 (J ≈ B); lr 5e-6 is slower. 24 tiles is clearly worse so far (−0.04, low recall).

### H11 — broken answers cost ~0.07 F1 (constrained decoding on the full fine-tune)
B ep5 gold by answer status: 35 valid pages F1 0.761; 12 schema-error pages 0.562; 3 unparsed pages 0 (261 gold
values). Full FT breaks the schema on 30% of pages (LoRA v6 constrained: 1 of 50). If constrained decoding fixes
them at near-valid quality, overall F1 → ~0.74–0.76; the broken pages may be harder, so less. Test: `truba/eval.slurm`
(new, scoring-only job) with `--constrained` on B ep5 (P_constr_B5) and J ep6 (P_constr_J6).

### 13:38 — daytime job limit; G, H, M, N, Q cancelled
User rule: at most 4 jobs running 08:00–20:00 (others use the cluster), no limit at night. Cancelled the runs that
test already-rejected ideas: G (24 tiles; last score validation ep5 0.674 = B's range, gold ep5 0.645 still −0.05),
H (lr 5e-6; ep7 gold 0.644), N (canonical labels seed 1; ep6 gold 0.665), M (combination, unscored), Q (vision seed 1,
just started), and P_constr_J6 (same question as P_constr_B5). L (vision encoder) ep5 validation 0.643 and ep6 gold
0.679: the ep5 gold 0.700 does not hold on validation, so training the encoder is not a gain so far.
Kept running: O_ft_half, P_constr_B5, L (last scoring), S_eval_b32 (scoring 32 pages per batch instead of 8: same
F1 and how much faster?). Full per-run table: `DEVLOG.md`.
S_eval_b32 (B ep5 gold, 32 pages per batch vs 8): lenient 0.685 vs 0.693; only 18/50 answers identical (left padding
+ bf16 change the numbers); generation 100 s vs 160 s. Decision: keep 8 per batch so all runs stay comparable;
batch composition is a second ~0.01 noise source. Most scoring time is batches waiting on a page that repeats to the
2,500-token limit.
**H11 result — rejected.** B ep5 gold with `--constrained`: lenient 0.694 vs 0.693 free; 49/50 valid. By free-decoding
status (strict F1, free → constrained): 35 valid pages 0.741 → 0.716; 12 schema-error pages 0.546 → 0.564 (all now
valid); 3 unparsed 0 → 0.499. The broken pages are hard pages: making them valid barely raises their content
score, and the constraint costs 0.025 on pages that were already valid. 5.4× slower (868 s vs 160 s). Keep
constrained decoding for deployment (valid JSON) but not as an F1 lever or for scoring experiments.
O_ft_half epoch 4: gold 0.610, validation 0.578 vs full data 0.683–0.686 / 0.660 → gap 0.075–0.08 and growing;
H10 (data size is the main limit) is very likely; waiting for its best epoch.

### H12 — more effective data without new labels: image augmentation (2026-10-13 ~14:30)
H10 evidence (half data −0.075 to −0.08 at epoch 4) says the 1B model is data-limited. Before paying for new teacher
labels, test whether random image noise on training pages acts as more data. New flag `train_lora.py --augment P`:
with probability P a page image is resized down (0.55–1.0) and back, rotated ±1.5°, contrast 0.7–1.2, brightness
0.85–1.1, blurred (30%, σ 0.3–0.8) and JPEG-compressed (q 45–90). Checked by eye on a training page: all text readable,
so labels stay true. Runs (one change each vs B): U_ft_aug50 (P 0.5), U_ft_aug50_seed1 (seed repeat), V_ft_aug50_8ep
(P 0.5, 8 epochs: augmentation should delay overfitting). Prediction: if augmentation adds data, U beats B by ≥0.02 on
both dev sets and V's best epoch is later than U's.
Also queued: Q_ft_vision_seed1 (vision encoder trained, seed 1: L was +0.007–0.014 on both sets, inside noise) and
R_ft_batch2 (2 pages per step, 1.45× faster in the 320-page test, 78.7 GB peak; must match B's F1 to be adopted).
**New-label sources measured:** hadikp = 8,905 one-page PDFs, image-only (no text layer), English resume template
samples (some with placeholders like "[Amount]"); embedded image width <900 px: 3,164; 900–1,199: 4,251; ≥1,200: 1,490.
Out of domain for the French/Tunisian dev pages; decision on labelling waits for O's best epoch and H12.
**O_ft_half final (H10):** gold ep3–6 0.613 0.610 0.621 0.641; validation ep3–6 0.572 0.578 0.592 0.584. Best gold 0.641 vs
full data 0.688–0.693 (−0.05), still rising at ep6. **Confound:** same 6 epochs over half the pages = half the
optimizer steps, so part of the gap may be undertraining, not fewer distinct pages. Test: O2_ft_half_12ep (half data,
12 epochs = B's step count; queued for 20:00, scored from ep8). If O2's best is still ≥0.03 below B on both sets,
distinct pages are the limit and new teacher labels are justified.
Constrained B ep5 validation: 0.676 vs 0.669 free (+0.007, inside noise) — H11 rejection stands.
**R_ft_batch2 (2 pages per step, accum 8; rejected):** gold ep4–6 0.686 0.669 0.664, validation 0.664 0.653 0.651.
Ties B at ep4, then falls instead of peaking at ep5 (B 0.693 / 0.669–0.677). Below B by ~0.02 on both sets at its best
→ keep 1 page per step despite the 1.45× speed. Likely cause unknown (padding inside a step changes the numbers).
O2_ft_half_12ep ep8 gold 0.647 (same number of steps as B ep4: 0.683).
**U_ft_aug50 (H12, image noise P 0.5):** gold ep4–6 0.691 0.695 **0.727**; validation 0.631 0.668 0.680. Gold +0.034–0.039
over B/I best (outside noise); validation +0.003–0.011 (inside noise). Still rising at ep6 while B falls after ep5, as
H12 predicts. Not yet "real" under the protocol (both sets); deciding runs: U_ft_aug50_seed1 and V_ft_aug50_8ep.
O2_ft_half_12ep: gold ep8 0.647, ep9 0.599; validation ep8 0.578 → with B's step count, half data is still 0.04–0.08
lower: distinct pages are the limit (H10 supported, confound removed).
O2 ep9 validation 0.573; cancelled after ep9 (falling on both sets, conclusion fixed). Started W_ft_aug100 (image noise on every page, P 1.0) in its slot.

### Decision (2026-10-13 ~15:00): image noise P 0.5 becomes the base recipe
Best epoch, lenient: U seed 42 gold 0.727 / validation 0.680; U seed 1 gold 0.720 (ep5) / validation 0.678 (ep5);
B / I gold 0.693 / 0.688, validation 0.669 / 0.677. Gold +0.027–0.039 across two seeds (outside noise); validation
+0.001–0.011 (inside noise, no loss). Under the protocol the gain is shown on gold only; adopted because it costs
nothing and does not lower validation. Running on top of it: V (8 epochs), W (P 1.0). Queued for 20:00: X (lr 2e-5),
Y (seed 2, spread of the new recipe), Z (+ vision encoder; combined candidate, only meaningful if Q shows a gain).
**More labels (H10 follow-up):** `src/prepare_hadikp.py` (new) copies the embedded JPEG of hadikp PDFs (width ≥900 px,
seeded sample, validation-hash ids skipped) to `data/hadikp/` and writes `documents_hadikp_*.jsonl` for
`auto_label.py --documents`. Pilot: 20 pages, 4 workers → `labels_hadikp_pilot.jsonl`; check by eye before scaling.
**Correction (~15:10):** U seed 1 ep6 gold 0.666 (ep5 0.720). Image-noise runs swing ±0.05 between epochs, so picking
the best epoch on the same 50 gold pages inflates them more than the steadier baseline. Mean gold ep4–6: U 0.704,
U seed 1 0.685, B 0.688, I 0.687 → gain +0.016 / −0.002, not established. "Adopted as base recipe" is withdrawn
until U seed 1 ep6 validation and Y (seed 2) are in. **Protocol change:** from now on choose the epoch on the
validation pages and report gold-50 at that epoch (and the mean over scored epochs), never the gold-best epoch.
U seed 1 ep6 validation 0.661 (ep5 0.678) → validation-chosen epoch 5, gold 0.720. With epochs chosen on validation:
U 0.727 / U seed 1 0.720 vs B 0.693 / I 0.686 on gold (+0.027–0.041); validation 0.680 / 0.678 vs 0.669 / 0.677
(level). Image-noise gain on gold survives unbiased epoch choice; third seed (Y, started now) decides adoption.

### 14:30 — vision encoder seed pair done (Q_ft_vision_seed1); verdict: undecided, third seed queued
- Q (train vision encoder, seed 1) lenient F1, ep4–6: gold 0.695 0.725 0.714; val 0.660 0.683 0.701.
  Epoch chosen on val = ep6: gold 0.714, val 0.701. Same-seed baseline I (val-chosen ep5): gold 0.686, val 0.677
  → +0.028 gold, +0.024 val (both above the 0.01–0.02 spread).
- Seed-42 pair: L (val-chosen ep6) gold 0.679, val 0.685 vs B (val-chosen ep5) gold 0.693, val 0.669 → −0.014 gold, +0.016 val.
- Means over ep4–6: gold L 0.687, Q 0.711 vs B 0.688, I 0.687.
- Reading: the seeds disagree on gold; val favours vision training in both pairs, but by little in the seed-42 pair.
  Not adopted yet. Third seed: AA_vision_seed2 (1602207, submitted 14:30). There is no baseline seed 2 yet, and both
  third-seed tests (Y image noise, AA vision) need one → submit AB_base_seed2 at the next free daytime slot.
- W (image noise on every page): ep4 gold 0.672 val 0.651; ep5 gold 0.717 val 0.668. ep6 pending.

### 14:50 — hadikp labelling pilot checked; 1,000-page labelling started
- Pilot (20 pages, 4 workers): 19 ok, 1 failed. Checked by eye against the page images:
  - hk-9d984fc02a43: name, phone, email, both employers, all 4 dates (09/2009 → 2009-09 …), highlights, education, certificate all match.
  - hk-3e8ebf241203: all 4 jobs' names, titles and dates match; the placeholder "City, STATE" is kept as printed (correct: label what the page shows).
  - The failed page hk-036fe79346c6 is broken in the source PDF: its anonymiser replaced every "t" with "Claire"
    ("SofClaireware"). The teacher refused it 6 times (2,135 s). Correct outcome, but slow.
  - No ok label contains the broken "Claire" text (regex over all 19), so the teacher did not copy or silently repair such pages.
- Started: documents_hadikp_1k.jsonl (1,000 pages, pilot pages and validation-split ids excluded) →
  labels_hadikp_1k.jsonl, 4 workers, --max-retries 3 (halves the time lost on broken pages). Started 14:47; expected 11–20 h.
- Next: when done, train baseline settings on train v6 + the ok hadikp labels (H10: more distinct pages help).

### 15:00 — W (image noise P 1.0) done: no gain over P 0.5; AB_base_seed2 submitted
- W val-chosen ep6: gold 0.705, val 0.680 vs U (P 0.5) ep6 0.727 / 0.680. More noise does not help; P stays 0.5.
- Submitted AB_base_seed2 (baseline, seed 2) as the same-seed reference for Y (image noise) and AA (vision encoder).

### Image noise (H12, --augment 0.5): adopted after 3 seeds — with a stated exception to the protocol
Epoch chosen on val, lenient F1 (gold / val):
| seed | image noise | baseline | diff |
|---|---|---|---|
| 42 | U 0.727 / 0.680 | B 0.693 / 0.669 | +0.034 / +0.011 |
| 1 | U1 0.720 / 0.678 | I 0.686 / 0.677 | +0.034 / +0.001 |
| 2 | Y 0.711 / 0.674 | AB 0.674 / 0.666 | +0.037 / +0.008 |
- Gold gain is consistent and above spread (+0.035 mean, 3/3). Val gain is positive 3/3 but small (+0.007 mean), below
  the 0.01–0.02 spread, so the "beats spread on BOTH sets" rule is NOT met. Adopted anyway because all 6 differences
  have the same sign, gold is the reviewed set the goal is measured on, and the change has no inference cost.
  Risk: the gold gain could partly be gold-specific (gold pages differ in look from the teacher-labelled val pages).
- P 1.0 (W) and 8 epochs (V) gave no further gain. New default: --augment 0.5, 6 epochs.
- Z1_aug_vision_seed1 moved from 20:00 to now (daytime slot freed by AB).

### Vision encoder training (--train-vision): adopted after 3 seeds
Epoch chosen on val, lenient F1 (gold / val), diff vs same-seed baseline:
seed 42 L vs B −0.014 / +0.016; seed 1 Q vs I +0.028 / +0.024; seed 2 AA vs AB +0.020 / +0.027.
- Val: above spread 3/3 (+0.022 mean). Gold: positive 2/3 (+0.011 mean, at the edge of spread). Partial pass on gold, adopted.
- Image noise and vision training each help mostly on a different dev set (noise: gold; vision: val).
  New default = both together (Z runs). Z (seed 42) running, Z1 (seed 1) and Z2 (seed 2) submitted;
  compare against the single-change runs U/U1/Y and L/Q/AA.

### X (image noise, lr 2e-5): highest val so far, gold mixed → seeds 1, 2 queued
- Val-chosen ep5: gold 0.705, val 0.717 vs U (lr 1e-5) 0.727 / 0.680 → −0.022 gold, +0.037 val. Gold mean ep4–6 0.709 vs U 0.704.
- Val is stable at ep5–6 (0.717, 0.714). Hypothesis: lr 1e-5 under-trains within 6 epochs; 2e-5 converges faster
  (ep3 val already 0.643 vs 0.628–0.632 for lr 1e-5 runs). Test: X1 (seed 1, vs U1), X2 (seed 2, vs Y, starts 20:00).

### Z (combined: image noise + vision encoder), seed 42: val 0.720 (best so far), gold 0.699
- vs B +0.006 gold / +0.051 val; vs U (noise only) −0.028 / +0.040; vs L (vision only) +0.020 / +0.035.
- Val gains look additive; gold does not. Z1, Z2 (seeds 1, 2) running. Submitted ZX_aug_vision_lr2e5 (Z + lr 2e-5, one change vs Z).

### Error analysis of the best run (Z_aug_vision ep5) — where the remaining F1 goes
Method: scratchpad `error_kinds.py` sorts each unmatched gold value (lenient form, same pairing as rescore.py) into
other_key (same text under another key), split_merge (substring of a predicted value at the same path), misread
(similar text at the same path, ratio ≥ 0.8), missing (none of these).
- Gold 1–50: 1,447 of 4,612 values unmatched: missing 869, split_merge 215, other_key 212, misread 151.
  Val: 2,121 of 7,203: missing 1,079, split_merge 462, other_key 309, misread 271. Reading errors are the smallest kind.
- Status split, gold: 40 valid pages F1 0.759; 7 schema errors (extra keys such as education.technologies,
  projects.location; invalid enum values) recall 0.64; **3 unparsed pages = 386 gold values (8%) scored 0**
  (2 hit the 2,500-token limit, one looping "2019-2019 | …"; 1 had one invalid \escape). Val: no unparsed page.
  One looping page moves gold F1 by ~0.03 → explains part of the ±0.05 epoch swings on gold.
- Not driven by page length: predicted/gold value count is 0.92–1.17 in every length third; recall 0.63–0.78
  throughout. "Missing" values are mostly replaced by different content: omitted items (dates, technologies),
  section headings used as skill names, teacher-inferred values not printed on the page ("~4/5" levels).
- Action: `rescore.py --repair` (new; writes rescore_repair.json; defaults unchanged): doubles invalid escapes and
  salvages cut-off answers (keeps complete values, closes brackets). Z ep5: gold 0.699 → 0.713, val 0.720 → 0.720;
  with --canon too: gold 0.731. Report --repair next to raw from now on; decisions stay on the default scores
  until all compared runs are rescored with --repair.
- Next hypotheses: (H13) more distinct pages (hadikp, labelling running) is still the main lever, since errors are
  spread over content kinds rather than one fixable kind. (H14, cheap) decoding loops: try repetition_penalty 1.05
  or a no-repeat n-gram limit on the looping pages only — measure on gold+val that valid pages do not get worse.

## 2026-10-13 17:40 — H15: reinforcement learning (GRPO) on top of supervised fine-tuning
- **Why:** supervised training rewards copying each label token; the metric rewards correct fields. 8% of gold values
  were lost on 3 pages whose answers were not valid JSON (loops to the 2,500-token limit, a bad escape). GRPO samples
  4 answers per page and pushes toward the higher-scoring ones, so it trains directly on the reported metric and
  should punish loops and broken JSON.
- **Setup:** `src/train_grpo.py` (TRL 1.15.0 GRPOTrainer from `$D/trl_pkgs`, a separate folder; the shared venv is
  unchanged). Reward = lenient field F1 of the answer vs the label, same pairing as `rescore.py`; not JSON → 0;
  schema error → scored on raw JSON minus 0.1. Checked on 5 labels: exact 1.0, half-cut 0.0, one renamed key 0.89.
  Start: Z_aug_vision checkpoint-528 (epoch 6: gold 0.705 / val 0.719). lr 1e-6, beta 0 (no reference model),
  4 samples per page, 4 pages per step. Validation CVs excluded. Processor loaded from the base model (max_patches 12).
- **First run:** G0_grpo_smoke (1602499), 8 pages = 2 steps, queued after X1 so the daytime limit of 4 holds.
  Checks: it runs with InternVL tiles, memory fits, seconds per step, and the saved model scores ≈ Z ep6.

## 2026-10-13 18:05 — Time use per job (measured on Z1_aug_vision_seed1)
- One job ≈ 65 min: training 33 min (6 epochs, 4.2 pages/s, per-device batch 1 × 16 accumulation), scoring ≈ 30 min
  (4 epochs × gold + val; 150 s + 190 s generation each plus 2 model loads).
- Scoring uses 6 GB of the 80 GB GPU at batch 8; a batch waits for its longest answer (looping pages run to 2,500 tokens).
- Epoch 3 was never the validation-chosen epoch in any run so far.
- Speed tests queued for 20:00: T_evalbatch32 (Z checkpoint-528 scored at batch 32 — must give the same F1 as
  batch 8, 0.705 / 0.719, before it is adopted) and ZB_batch4_speed (Z settings with batch 4 × accumulation 4, same
  effective batch; compare minutes and F1 to Z; MIN_EPOCH 4).

## 2026-10-13 18:35 — Combined setting (image noise + vision encoder) over 3 seeds
Val-chosen epoch, lenient F1, gold / val, mean of seeds 42, 1, 2:

| Setting | seed 42 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| baseline full FT (B, I, AB) | 0.693 / 0.669 | 0.686 / 0.677 | 0.674 / 0.666 | 0.684 / 0.671 |
| image noise P 0.5 (U, U1, Y) | 0.727 / 0.680 | 0.720 / 0.678 | 0.711 / 0.674 | 0.719 / 0.677 |
| vision encoder trained (L, Q, AA) | 0.679 / 0.685 | 0.714 / 0.701 | 0.694 / 0.693 | 0.696 / 0.693 |
| combined (Z, Z1, Z2) | 0.699 / 0.720 | 0.722 / 0.711 | 0.697 / 0.693 | 0.706 / 0.708 |

- Combined vs baseline: +0.022 gold, +0.037 val — above the 0.01–0.02 spread on both sets. **Adopted as the default.**
- Combined vs image noise only: −0.013 gold, +0.031 val. Combined vs vision only: +0.010 / +0.015. The two dev sets
  disagree on image noise alone vs combined; combined is the only setting that is high on both.
- lr 2e-5 (X 0.705 / 0.717, X1 0.714 / 0.695): X1 val 0.022 below X; waiting for X2 before a decision.

## 2026-10-13 18:50 — GRPO smoke test passed (G0_grpo_smoke, 1602515)
- First attempt (1602499) failed before training: dill pickled the TRL-folder modules by value because the folder was
  outside the venv prefix (PicklingError MonthDayNano in datasets fingerprinting). Moved to `$D/venv/trl_pkgs`.
- 2 steps × 4 pages × 4 samples: 20–33 s per step, no out-of-memory. Sampled answers 505–1,341 tokens, none cut at the
  limit. Reward mean 0.70, std 0.22–0.28 within groups, no group with all-equal rewards — the reward separates samples.
- eos token after TRL's config alignment is 151645, the same as the SFT checkpoint.
- Queued for 20:00 from Z checkpoint-528, one epoch over ~1,390 training pages (≈ 350 steps, ≈ 2.5–3 h):
  G1_grpo_lr1e-6 (1602517) and G1_grpo_lr3e-6 (1602518). Baseline: Z ep6 0.705 / 0.719.

## 2026-10-13 19:10 — Scoring batch 32 vs 8 (T_evalbatch32, 1602543): not adopted
- Z checkpoint-528: gold 0.707 (batch 8: 0.705), val 0.717 (0.719). Gold took 192 s with model loading vs 178 s of
  generation at batch 8: no speed gain — each batch still waits for its longest answer.
- 33 of 50 gold answers differ in text between batch 8 and 32 (bf16 greedy decoding depends on padding). F1 moved
  ≤ 0.002, so this is a small noise source; all runs keep scoring at batch 8 so they stay comparable.
- G0 GRPO smoke final model: gold 0.721 / val 0.721 (start 0.705 / 0.719) after 2 steps — the pipeline works.

## 2026-10-13 19:25 — Training batch > 1: does not fit, dropped
- Batch 4 × accum 4 (1602568) and batch 2 × accum 8 (1602610), no gradient checkpointing, 12 tiles: both out of GPU
  memory (79 GB) within the first 3 steps; the long pages (≈ 4,500 tokens with 3,072 image tokens) need > 70 GB.
- Batch 1 already processes ≈ 4,500 tokens per forward pass at 4.2 pages/s; turning on gradient checkpointing to fit
  a larger batch adds ≈ 30 % recompute. Training stays at batch 1 × accum 16.

## 2026-10-13 19:35 — H14 loop control (scoring only) and third seed for extra pages, queued for 20:00
- `evaluate.py --repetition-penalty` added (default 1.0 = unchanged). R_reppen1.02 (1602623) and R_reppen1.05
  (1602624) score Z checkpoint-528 (0.705 / 0.719 at 1.0). Risk: JSON repeats keys on purpose, so the penalty can
  also lower tokens that should repeat; no-repeat-n-gram was not tried for that reason (it would block repeated keys).
  Success: fewer pages at the 2,500-token limit and F1 up on both dev sets.
- ZH2_hk241_seed2 (1602625): third seed for the 241 extra pages (H13), compared against Z, Z1, Z2.

## 2026-10-13 19:15 — H14 loop control: repetition penalty 1.05 (R_reppen1.05, 1602677) — not adopted
- Z checkpoint-528, batch 8: gold 0.712 (no penalty 0.705), val 0.705 (0.719). No answer hit the 2,500-token limit and
  0 gold answers were unparsed, but val lost 0.014 — the penalty also lowers tokens that should repeat (JSON keys,
  repeated dates). Mixed result across the two dev sets → not adopted. 1.02 running (1602688).
- Monitor fix: score lines from ad-hoc jobs (`runs/<name>/<set>:`) are now collected too.
- Repetition penalty 1.02 (R_reppen1.02, 1602688): gold 0.706 / val 0.711 (no penalty 0.705 / 0.719). No gain.
  **H14 closed for decoding-time penalties:** loops are better removed in training (GRPO scores a looping answer 0).

## 2026-10-13 19:30 — GRPO sampling temperature (queued for 20:00)
- G2_grpo_temp0.7: G1_grpo_lr1e-6 with temperature 0.7 instead of 1.0 (one change). Why: at 1.0 the sampled answers
  scored 0.70 on training pages, well below greedy decoding; samples closer to the greedy answer may give a more
  useful comparison inside each group of 4. Risk: less spread inside a group → weaker learning signal (watch
  `frac_reward_zero_std` and `reward_std` in log_history).
- 24 tiles was not re-queued: already rejected for the 1B model (G_ft_24tiles, −0.05 gold).

## 2026-10-13 19:35 — lr 2e-5 with image noise over 3 seeds: not adopted
- X / X1 / X2 (val-chosen): 0.705 / 0.717, 0.714 / 0.695, 0.713 / 0.699 → mean 0.711 / 0.704.
- vs image noise at lr 1e-5 (mean 0.719 / 0.677): gold −0.008 (within spread), val +0.027.
- vs the combined default (image noise + vision encoder, lr 1e-5; mean 0.706 / 0.708): +0.005 / −0.004 — the same.
  Combined at lr 2e-5 (ZX 0.704 / 0.705) gave no gain either. **Default stays: combined at lr 1e-5.**

## 2026-10-13 19:55 — Where the lost values are (Z ep5 gold, lenient F1 per section, weighted by gold values)
| Section | gold values | F1 | ≈ values lost |
|---|---|---|---|
| projects | 829 | 0.648 | 292 |
| skills | 1,023 | 0.717 | 290 |
| work | 1,068 | 0.750 | 267 |
| education | 566 | 0.617 | 217 |
| basics | 391 | 0.810 | 74 |
| certificates | 146 | 0.603 | 58 |
- Education is the lowest large section; projects is second (and projects vs work was the main other_key confusion).
  Next check (no GPU): per-section agreement of the two teacher runs (H0). If education/projects are also low there,
  the labels are ambiguous in those sections and the fix is a labelling rule, not more training.
- Z3_aug_vision_seed3 (1602713) queued for 20:00: a 4th seed of the default, to narrow the baseline spread.
- **Answer (from H0, already measured):** teacher run 2 vs gold per section: education 0.89, projects 0.91, skills
  0.91, work 0.94. Student gaps to that ceiling: education −0.27, projects −0.26, skills −0.19, work −0.19. The
  labels are consistent enough in these sections; the loss is the student's, not label ambiguity. Education and
  projects are the sections to inspect page by page next (what is missing: whole entries, or fields inside entries).
- **Entries vs fields (Z ep5 gold):** entry counts are close to gold (education 117 vs 109, projects 116 vs 117,
  work 110 vs 127); whole sections are missing only on the 3 unparsed pages. The loss is inside entries.
- Largest field losses (parsed pages, lenient): skills.keywords 168 (R 0.76), work.highlights 51 (R 0.62),
  work.technologies 49, projects.technologies 45, education.studyType 42 (R 0.49), projects.name 38,
  projects.highlights 35 (R 0.36), projects.description 29 (R 0.40), work.summary 22 (R 0.55).
- Two patterns: (1) long sentence fields (highlights, description, summary) score low because one wrong letter
  anywhere fails the whole value, and the model also swaps text between highlights / description / summary;
  (2) education.studyType vs area is split differently ("Bachelor of Science in X"). GRPO's reward counts exactly
  these, so it is the right lever to test first; a per-field look at its output comes after its scores.

## 2026-10-13 20:05 — GRPO with a KL term (G3_grpo_beta0.04, queued)
- G1_grpo_lr1e-6 with beta 0.04 instead of 0 (one change): a penalty for moving away from the starting model. Why:
  with beta 0 nothing stops the policy from drifting on fields the reward rarely sees; the KL term keeps the
  output format of the supervised model. Cost: a second (reference) copy of the model in memory, ~2 GB.

## 2026-10-13 20:15 — H13 (241 extra hadikp pages, +16 % training pages) over 3 seeds: no clear gain
| seed | with extra pages | without (Z, Z1, Z2) | difference |
|---|---|---|---|
| 42 | 0.697 / 0.720 | 0.699 / 0.720 | −0.002 / 0.000 |
| 1 | 0.725 / 0.710 | 0.722 / 0.711 | +0.003 / −0.001 |
| 2 | 0.722 / 0.710 | 0.697 / 0.693 | +0.025 / +0.017 |
| mean | 0.715 / 0.713 | 0.706 / 0.708 | +0.009 / +0.005 |
- Within the 0.01–0.02 spread; 2 of 3 seeds show no change. Not adopted yet. Labelling continues (≈ 100 pages/h);
  re-test when ≈ 1,000 extra pages are labelled (+70 %), which is a large enough change to see an effect.

## 2026-10-13 21:00 — H15 first result: GRPO lr 1e-6 (G1_grpo_lr1e-6) — no measurable change
- gold 0.711 / val 0.718 vs start 0.705 / 0.719: within spread on both sets.
- The sampled answers' reward rose (0.73 → ~0.77) but greedy decoding on the dev pages barely moved: at lr 1e-6 for
  ≈ 350 steps the weights changed too little to change the greedy answer. Gold had fewer broken answers
  (unparsed 3 → 1, limit hits 2 → 0), validation one more (2 → 3) — no clear effect on loops either.
- lr 3e-6 (running, step 186: training reward 0.80 at steps 101–150) is the next test; if it also shows no dev change,
  GRPO with this reward needs either many more steps or a reward that does not saturate on easy pages.

## 2026-10-13 21:30 — Default (combined) over 4 seeds: the reference for every later comparison
- Val-chosen: 0.699 / 0.720, 0.722 / 0.711, 0.697 / 0.693, 0.699 / 0.708 → **mean 0.704 / 0.708**;
  range gold 0.697–0.722, val 0.693–0.720 (standard deviation ≈ 0.012 on each set).
- A single run must differ from 0.704 / 0.708 by more than ≈ 0.025 to be outside this range; a 3-seed mean by ≈ 0.014.

## 2026-10-13 22:00 — H15 second result: GRPO lr 3e-6 (G1_grpo_lr3e-6) — no gain
- gold 0.718 / val 0.704 vs start 0.705 / 0.719: +0.013 gold, −0.015 val. Opposite signs, both within the spread.
- Training reward rose 0.73 → 0.80 in the first 150 steps and then stayed at ~0.78 (steps 200–321). At 3× the lr the
  weights did move, but the dev score did not. GRPO with this reward is not adopted. G2 (temperature 0.7) and
  G3 (KL 0.04) finish tonight and close the question; no further GRPO runs unless one of them beats the spread.

## 2026-10-13 22:05 — H16: average the weights of the 4 default seeds (model soup), S_soup4_default (1602956)
- Hypothesis: the 4 seeds start from the same base weights and move little (lr 1e-5, 6 epochs), so their weights
  can be averaged; the average keeps what all seeds learned and cancels part of each seed's own errors
  (Wortsman et al. 2022, arXiv 2203.05482: uniform soups of fine-tunes from one start). No training: ~25 min of scoring.
- Members (kept weights; Z2 kept ep4 under the older gold-chosen rule): Z ep6 0.705 / 0.719, Z1 ep6 0.722 / 0.711,
  Z2 ep4 0.703 / 0.686, Z3 ep5 0.699 / 0.708 → members' mean 0.707 / 0.706, best member 0.722 / 0.711.
- Success: above the members' mean by more than ~0.014 on both sets. Failure sign: a broken answer format (the
  averaged weights fall between the seeds' solutions) → many unparsed pages.
- Tools: `src/soup.py` (averages model.safetensors of N folders, checked on two fake 2-value tensors: 1 and 3 → 2),
  `truba/soup.slurm` (average, then score gold-50 + val with --batch 8).

## 2026-10-13 22:30 — H15 third result: GRPO temperature 0.7 (G2_grpo_temp0.7) — no change
- gold 0.705 / val 0.714 vs start 0.705 / 0.719. Lower temperature raised the training reward (~0.85) only because
  the samples are closer to the greedy answer; the greedy answers on dev pages did not change.
- **Result (22:15):** gold 0.721 / val 0.711 vs members' mean 0.707 / 0.706: +0.014 / +0.005. Gold at the threshold,
  val within spread → not a clear gain, but the average equals the best single member on both sets without choosing a
  seed, and the answer format did not break. Precision 0.758 > recall 0.688 on gold.
- Next (queued 22:15, 10 min each): more members. S_soup10_vision = 10 models trained with the vision encoder
  (Z/Z1/Z2 ep4+ep6/Z3/ZH/ZH1/ZH2/ZX ep5+ep6); S_soup16_wide = those + 6 trained with image noise only, vision
  frozen (X, X1, X2, U, U1, Y). Averages tend to gain with more good members; if 16 > 10 > 4, averaging is adopted.

## 2026-10-13 22:35 — H16 result: weight averaging ADOPTED
| model | gold | val |
|---|---|---|
| default, mean of 4 seeds | 0.704 | 0.708 |
| average of the 4 default seeds | 0.721 | 0.711 |
| average of 10 vision-trained models (S_soup10_vision) | 0.728 | 0.732 |
| average of 16 models (+6 vision-frozen, S_soup16_wide) | 0.744 | 0.723 |
- Both large averages beat the default by more than the spread on both sets (+0.024 / +0.024 and +0.040 / +0.015).
  More members helped: 4 → 10 → 16 raised gold each time. Val prefers 10 (vision-trained only); by protocol (choose on
  val) the 10-model average is the current best, and gold for it is 0.728. The 16-model one is the gold-best (0.744).
- Why it works (likely): each seed makes different reading mistakes; averaging the weights keeps what they share.
  Recall rose more than precision (16-model: R 0.732 vs 4-model R 0.688), i.e. fewer values left out.
- Members are now the product: every further training run is also a candidate member. Queued 3 more default seeds
  (Z4, Z5, Z6; 1602988–90). At ~03:00 the 1k hadikp labels finish → seeds trained with the extra pages (H13 re-test)
  also become members. Test set (gold 51–254) for the best average after that.
- Disk: averages and their members are full-precision (3.75 GB each). Deleted GRPO models (G0, G1 ×2, G2; GRPO not
  adopted) and the 4-model average's weights.

## 2026-10-13 23:00 — SmolVLM-500M baseline (download approved by the user 2026-10-13)
- Downloaded HuggingFaceTB/SmolVLM-500M-Instruct to TRUBA hf_cache (0.97 GB; its 4.7 GB ONNX folder deleted).
- `src/train_lora.py` / `src/evaluate.py` now pick per-model settings from the processor class (FAMILY table):
  answer header ("<|im_start|>assistant\n" vs "\nAssistant:"), end token (<|im_end|> vs <end_of_utterance>),
  vision-encoder prefix (model.vision_tower vs model.vision_model), image size (InternVL max_patches 12; SmolVLM
  longest_edge 2048 = 12 tiles of 512 px + thumbnail; other values refused). Perturbation range = tokenizer.vocab_size
  (= 151,643 for InternVL, the old <|endoftext|> id, so InternVL runs are unchanged).
- Checked on 2 real training pages (TRUBA CPU): SmolVLM image tokens 64 per tile, all masked; answer starts at
  ' {"basics":' and ends with <end_of_utterance>; InternVL unchanged (16 tiles × 256 masked, same header/end).
  SmolVLM tokens per answer are ~20 % more (median 1,290 vs 1,084 on 40 pages, max 1,964) → scored with
  --max-new-tokens 3000. SmolVLM sees 832 image tokens per page vs InternVL's 3,328.
- Queued: SV_smolvlm_default (1602995) = the default recipe with only the model changed; T_evalcheck_family (1602997)
  = scoring regression (Z3 ep5 on 8 gold pages must match its earlier answers) + SmolVLM base on 8 gold pages.

## 2026-10-13 23:30 — H15 closed: GRPO with the field-F1 reward does not change dev scores
| run | change | gold | val |
|---|---|---|---|
| start (Z ep6) | — | 0.705 | 0.719 |
| G1 | lr 1e-6 | 0.711 | 0.718 |
| G1 | lr 3e-6 | 0.718 | 0.704 |
| G2 | temperature 0.7 | 0.705 | 0.714 |
| G3 | KL beta 0.04 | 0.707 | 0.715 |
- All within the spread. Training reward rose in each run (0.73 → 0.78–0.85) but greedy answers on new pages did not
  improve: ~350 steps of 4 pages × 4 samples learn the training pages, not reading. Weight averaging (H16) gave more.
  Not re-opened unless a new reward idea addresses reading errors specifically.
- **Check result (T_evalcheck_family, 23:35):** new scoring code on Z3 ep5, 8 gold pages: 8 of 8 answers identical
  to the earlier run → InternVL scoring unchanged. SmolVLM base (untrained): runs with batch 8; 0 of 8 parsed (7 hit
  the 3,000-token limit) — it reads the page text but writes its own JSON layout, as expected before training.

## 2026-10-14 00:10 — Cleanup (user: obsolete things need not be kept)
- Deleted weights of runs that are neither average members nor still compared: AA, AB, B, I, L, Q (full FT), the
  out-of-scope C (2B), D and F (8B) LoRA adapters, old LoRA A_v6_seed1, lora_v6, E_dpo_v6; unused extra checkpoints
  of U1, X, X1; empty run folders; probe scripts. ≈ 60 GB. Scores, eval folders and logs kept.


## 2026-10-14 00:20 — More seeds of the same recipe do not raise the average further
| average | members | gold | val |
|---|---|---|---|
| S_soup10_vision | 10 vision-trained | 0.728 | 0.732 |
| S_soup13_vision | + Z4, Z5, Z6 | 0.725 | 0.730 |
| S_soup16_wide | 10 + 6 vision-frozen | 0.744 | 0.723 |
| S_soup19_wide | 13 + 6 vision-frozen | 0.745 | 0.724 |
- Adding 3 seeds changed nothing (±0.003). The average stops gaining at ~10 members of one recipe: the seeds' shared
  errors stay. Members trained differently (vision-frozen) moved gold (+0.017) but not val; differences between the
  10/16 averages are within the spread on each set. Current best by protocol (val): S_soup10_vision. Weights of the
  13, 16 and 19 averages deleted (rebuildable from members in ~10 min).
- Next: members that differ in what they were trained on, not in seed: the extra hadikp pages (1k batch finishes
  ~02:30). Z7/Z8 (running) finish as planned but are not expected to change the average.

## 2026-10-14 00:35 — SmolVLM track: its own loop (user: both tracks are improved, not one baseline)
Starting point SV_smolvlm_default (InternVL default recipe, model swapped): lenient gold / val ep3 0.300 / 0.345,
ep4 0.396 / 0.418, ep5 0.374 / 0.437 — still rising at ep5. Base reading was weak (0.16 of page words, zero-shot).
Three one-change runs against SV_smolvlm_default, queued 00:35:
| run | change | why |
|---|---|---|
| SV_lr3e-5 (1603212) | lr 1e-5 → 3e-5 | 0.5 B model still improving at ep5–6: likely under-trained at the 1B model's lr |
| SV_ep10 (1603214) | 6 → 10 epochs (scored from ep6) | same reason, other lever: longer instead of larger steps |
| SV_tiles30 (1603213) | 12 → 30 tiles (longest_edge 2048 → 3072 px) | it gets 64 tokens per 512 px tile (InternVL 256 per 448 px), i.e. ~4× less image detail per area; reading errors should fall with more tiles |
- Code: `--max-patches` for SmolVLM now selects longest_edge 2048 / 2560 / 3072 (12 / 20 / 30 tiles on A4; checked on
  5 training pages: 13 / 21 / 31 tiles incl. thumbnail = 832 / 1,344 / 1,984 image tokens). Other values stop the run.
- **SV_smolvlm_default final:** val-chosen ep6 gold 0.370 / val 0.460 (val still rising; gold fell after ep4).
  Main loss found: **repetition loops** — 10 of 50 gold pages hit the 3,000-token limit, every one ends in a loop
  (last 600 characters: 0–9 distinct strings out of 27–74); 14 of 79 val pages unparsed. Those pages score 0, so loops
  alone cost roughly 0.2 of recall. InternVL loops on ~2 of 50.
- Added (scoring only, `truba/eval.slurm` = score one saved model with extra args): SV_reppen1.05 (1603219) and
  SV_reppen1.1 (1603220) on the ep6 model. Repetition penalty did nothing for InternVL (H14), but there loops were
  rare; here they are a fifth of pages. Success: fewer limit hits and higher F1 on both sets.

## 2026-10-14 01:30 — Where the best InternVL average still loses values; answer repair v2 (adopted as post-processing)
- S_soup10_vision gold, lost gold values by kind (`scratchpad/error_kinds.py`): 835 "missing" (no similar value at the
  same field), 202 other field, 195 split/merge, 139 misread; 1,050 predicted values not in gold. Of the 835 missing,
  **308 are on 3 unparsed pages** (6 % of pages, 6.7 % of all gold values): one loop to the token limit, one invalid
  \escape, one missing comma mid-answer. On parsed pages the remaining 527 are mostly values the model did not write
  at that field at all (319) — omissions and different wording, spread over all sections.
- The old `rescore.py --repair` (salvage a cut-off answer) made gold worse (0.728 → 0.685): a salvaged loop page adds
  ~700 repeated values. **Repair v2** (now `--repair`): fix escapes; else keep complete values up to the cut-off end
  or, failing that, up to the first syntax error; then drop exact duplicate list items. Valid answers untouched.
| eval (lenient) | raw | old repair | repair v2 |
|---|---|---|---|
| S_soup10_vision gold / val | 0.728 / 0.732 | 0.685 / 0.732 | **0.746** / 0.732 |
| S_soup16_wide gold / val | 0.744 / 0.723 | 0.731 / 0.726 | 0.753 / 0.726 |
| Z ep5 / Z1 ep6 / Z6 ep6 gold | 0.699 / 0.722 / 0.737 | 0.713 / 0.690 / 0.677 | 0.713 / 0.734 / 0.736 |
| Z ep5 / Z1 ep6 / Z6 ep6 val | 0.720 / 0.711 / 0.712 | 0.720 / 0.701 / 0.700 | 0.720 / 0.715 / 0.719 |
| SV_smolvlm_default ep6 gold / val | 0.370 / 0.460 | 0.294 / 0.377 | 0.386 / 0.466 |
- Repair v2 never lowered a score by more than 0.001 and raised it up to +0.018; it is part of the product (code turns
  model text into JSON). From now on reported next to raw; run decisions still compare raw with raw (same footing).
- Dropping duplicates on every answer (not only broken ones) was also tried: same on gold, −0.002 to −0.003 on val
  for two runs → not used.

## H17 (InternVL): more teacher-labelled pages — 2,465 instead of 1,634 (2026-10-14 02:40)
- **Why:** the first test of extra pages (ZH runs, +241 pages) was within run-to-run spread, but those runs
  were good soup members. The 1k labelling batch is done (974 ok / 26 not ok), so the training file can grow by 993
  pages (+67% over the 1,472 base pages), enough that a real effect should exceed the 0.01–0.02 spread.
- **File:** `data/processed/labels_train_v6_hk1k.jsonl` = `labels_train_v6.jsonl` (1,472) + 993 `status == ok` pages from
  `labels_hadikp_pilot.jsonl` and `labels_hadikp_1k.jsonl`. Checked: all records pass `schema.to_record`, no
  duplicate (cv_id, page), 0 pages in the validation split, all images present on TRUBA.
- **Runs:** ZK_hk1k_seed42 / seed1 / seed2 (jobs 1603400–1603402), default recipe + this file, scored ep3–6.
- **Compare against:** default recipe 9-seed mean 0.705 / 0.708 (gold / val, lenient, val-chosen epoch).
- **Decision rule:** adopt if the 3-seed mean beats 0.705 / 0.708 by more than 0.02 on both sets. Either way, try
  the runs as extra members of the 10-model average (best so far 0.728 / 0.732).

## SmolVLM: repetition penalty at generation time — rejected (2026-10-14 ~04:15)
- **Hypothesis:** SmolVLM loses most of its F1 on answers that loop until the token limit (14 of 79 val pages
  unparsed). A repetition penalty while generating should stop the loops without retraining.
- **Runs (scoring only, weights of SV_smolvlm_default ep6):**

| Penalty | Gold lenient | Val lenient | Unparsed gold / val (hand count) |
|---|---|---|---|
| none (baseline) | 0.370 | 0.460 | 11 / 14 |
| 1.05 (SV_reppen1.05) | 0.342 | 0.430 | 9 / 10 |
| 1.1 (SV_reppen1.1) | 0.370 | 0.407 | — |

- **Result:** fewer loops, but lower F1 on both sets (1.05: −0.03 / −0.03; 1.1: 0 / −0.05). CVs repeat tokens on
  purpose (digits in dates and phones, repeated words in titles); the penalty pushes the model off correct copies.
- **Next (SmolVLM):** fix loops through training data instead. SV_hk1k: baseline recipe + the 2,465-page file
  (`labels_train_v6_hk1k.jsonl`), the same change as InternVL H17, so both tracks test more data.

## SmolVLM: learning rate 3e-5 instead of 1e-5 — adopted (2026-10-14 ~04:50)
- **Hypothesis:** 1e-5 was copied from InternVL (0.94 B); a smaller model with fewer steps per field may be under-trained.
- **SV_lr3e-5** (one change vs SV_smolvlm_default), lenient F1 gold / val per epoch:
  ep3 0.431/0.436, ep4 0.519/0.508, ep5 0.513/0.542, ep6 0.536/0.521. Val-chosen ep5: **0.513 / 0.542**
  vs baseline 0.370 / 0.460 (+0.14 / +0.08, far above the 0.01–0.02 spread; single seed).
- Unparsed pages at ep5: gold 3 of 50 (baseline 11), val 9 of 79 (baseline 14). Schema errors (valid JSON, wrong
  shape, still scored): gold 11, val 8 — the next failure type to inspect.
- **New SmolVLM default lr 3e-5.** Next: SV_lr6e-5 and SV_lr1e-4 (does the trend continue?), and the extra-pages run
  resubmitted at lr 3e-5 (SV_hk1k_lr3e-5; the queued lr-1e-5 SV_hk1k was cancelled before it started).
  SV_tiles30 and SV_ep10 (still at lr 1e-5) are compared against the old baseline.

## SmolVLM: 30 image tiles instead of 12 (lr 1e-5) — not shown (2026-10-14 ~05:00)
- **Hypothesis:** SmolVLM sees 64 tokens per 512-px tile, so 12 tiles may be too coarse to read small CV text.
- **SV_tiles30** vs SV_smolvlm_default, lenient gold / val. ep3–6: 0.306/0.388, 0.428/0.456, 0.399/0.439, 0.423/0.448.
  Val-chosen ep4 **0.428 / 0.456** vs 0.370 / 0.460: gold +0.06, val −0.004 → not real on both sets. Mean over
  epochs 0.389 / 0.433 vs 0.360 / 0.415 (small gain on both).
- **Next:** repeat on the new default (SV_lr3e-5_tiles30, lr 3e-5 + 30 tiles, one change vs SV_lr3e-5); at the
  better learning rate the model may use the extra detail. Weights deleted.

## SmolVLM: 10 epochs instead of 6 (lr 1e-5) — real gain, but below lr 3e-5 (2026-10-14 ~05:15)
- **SV_ep10** vs SV_smolvlm_default, lenient gold / val, ep6–10: 0.424/0.443, 0.420/0.465, 0.416/0.492, 0.462/0.487,
  0.419/0.485. Val-chosen ep8 **0.416 / 0.492** vs 0.370 / 0.460 (+0.05 / +0.03, real on both sets).
- **Reading:** together with SV_lr3e-5 (0.513 / 0.542 in 6 epochs), the lr-1e-5 SmolVLM was under-trained; more
  steps help, a larger learning rate helps more. Not adopted (lr 3e-5 is better). Weights deleted.
- **Next:** wait for SV_lr6e-5 / SV_lr1e-4; if the best rate still peaks at the last epoch, test more epochs at that rate.

### H17 result (2026-10-14 ~05:55) — extra pages adopted for InternVL, with one caveat
| Run | Val-chosen epoch | Gold | Val |
|---|---|---|---|
| ZK_hk1k_seed42 | 6 | 0.723 | 0.722 |
| ZK_hk1k_seed1 | 5 | 0.729 | 0.713 |
| ZK_hk1k_seed2 | 6 | 0.734 | 0.740 |
| **mean of 3** | | **0.729** | **0.725** |
| default recipe, mean of 9 seeds | | 0.705 | 0.708 |

- Gain +0.024 gold / +0.017 val. The val gain is just under the 0.02 threshold I set in advance; but every one of the
  3 runs beats the 9-seed mean on both sets, and the 9 baseline val scores span only 0.693–0.720 (two of three ZK
  runs are at or above the top of that range). Adopted as the InternVL training file from now on; caveat recorded.
- All three runs score best at epoch 5–6 (the last two): more data may want more epochs — a later test.
- **Next:** S_soup13k (best 10-model average + the 3 ZK models), S_soup3k (the 3 ZK models only), and ZK seeds 3–5
  so that an average of extra-page models alone can grow to ~6+ members.
- Deleted the 6 vision-frozen models (X, X1, X2, U, U1, Y) used only by the rejected 16- and 19-model averages.

## SmolVLM: learning rate 6e-5 — adopted (2026-10-14 ~07:55)
- **SV_lr6e-5** (one change vs SV_lr3e-5), lenient gold / val ep3–6: 0.487/0.503, 0.551/0.566, 0.550/0.558, 0.545/0.589.
  Val-chosen ep6 **0.545 / 0.589** vs 0.513 / 0.542 (+0.03 / +0.05, real on both). Single seed.
- Learning-rate trend so far: 1e-5 → 0.370/0.460, 3e-5 → 0.513/0.542, 6e-5 → 0.545/0.589. New SmolVLM default lr 6e-5.
- Val is highest at the last epoch → **SV_lr6e-5_ep10** (10 epochs, scored ep6–10), queued. SV_lr1e-4 (20:00) tells
  whether the rate can go higher. SV_hk1k_lr3e-5 and SV_lr3e-5_tiles30 are compared with SV_lr3e-5 (same lr).
- Deleted SV_lr3e-5 weights (scores kept).

## SmolVLM: 2,465 training pages instead of 1,472 (lr 3e-5) — not adopted (2026-10-14 ~09:20)
- **SV_hk1k_lr3e-5** vs SV_lr3e-5 (same lr), lenient gold / val ep3–6: 0.446/0.489, 0.457/0.456, 0.495/0.518, 0.519/0.501.
  Val-chosen ep5 **0.495 / 0.518** vs 0.513 / 0.542 (−0.02 / −0.02; single seed, about the size of the spread).
- **Reading:** unlike InternVL (+0.02), extra pages do not help SmolVLM. Consistent with SmolVLM being limited by
  reading the page, not by the amount of JSON examples. Weights deleted.

## 2026-10-14 ~09:40 — Literature review (user: "stuck with simple solutions"); research hypotheses
- Gap acknowledged: since the loop fix, runs were mostly tuning (+0.01–0.03 each). Rule added to the project rules: each track
  always has a literature-backed hypothesis queued or in preparation.
- Review by a Sonnet research agent: `literature/literature_review_2026-10-14.md` (11 ranked hypotheses, sources).
- Spot-checked by me: Pix2Struct (2210.03347) ablation, DocVQA: full pretraining 67.8, without the reading warm-up 56.2,
  without screenshot parsing (reading only) 12.2 — read in the PDF. Caveat: that is pretraining from scratch, not a
  stage added to an already-trained VLM; transfer to our case is inferred. Alibaba resume parser (2510.09722): abstract
  confirms a 0.6B model with parallel sub-task prompts; the 0.964 F1 is not in the abstract (not checked further).
- Flaw found in hypothesis 4 as written: `no_repeat_ngram_size=4` would forbid JSON's own repeated 4-grams
  (`"},{"name":"` repeats on every list item). A retry of failed pages must use sampling or a penalty, not n-gram bans.
- Plan: (a) SmolVLM reading stage (needs page transcripts: granite-docling on TRUBA needs approval, or GLM teacher);
  (b) section-by-section JSON generation (no approval needed); (c) retry of failed pages only (no training);
  (d) granite-docling-258M as a third starting model (needs approval); (e) synthetic rendered CV pages (later).

## 2026-10-14 ~10:30 — Owner decisions; granite-docling track; content score; two InternVL averages
- **Owner approved (2026-10-14):** granite-docling-258M download to TRUBA (done; 12 files), for page transcripts AND as a
  third student track (run prefix GD_); a content score next to the current score.
- **Code:** `src/train_lora.py` — granite-docling chat format (told apart from SmolVLM by its chat template; header
  `<|start_of_role|>assistant<|end_of_role|>`, end `<|end_of_text|>`); rows with a `transcript` key train the reading
  task (prompt "Write out all text on this page."); token perturbation never inserts a special token (granite counts
  its <image> token inside vocab_size). Mask check on TRUBA for all three models: answer and end token learnt, image
  tokens masked. `src/evaluate.py` — checkpoints without a processor: granite vs SmolVLM by vocabulary size
  (100,352 vs 49,280). `src/transcribe.py` + `truba/transcribe.slurm` — granite-docling DocTags → plain text on
  training pages only (stops if a page belongs to a gold CV; validation pages skipped).
- **Queued:** TR_test (transcripts of 16 pages: check speed and Turkish/French accents), GD_lr6e-5 (first granite run,
  SmolVLM's best recipe).
- **Content score** (`rescore.py`, printed after lenient): words of all values of a section, per page, as a multiset.
  Ignores how text is split into values and which field of the same section holds it; any letter/digit difference
  in a word is a miss. Caveat: it also ignores word order inside a section. Not yet measured: the teacher-vs-teacher
  content F1 (the ceiling under this score).

| Result | lenient gold / val | content gold / val | content, repaired, gold / val |
|---|---|---|---|
| S_soup10_vision (best InternVL) | 0.728 / 0.732 | 0.839 / 0.852 | 0.865 / 0.852 |
| S_soup13k (soup10 + 3 extra-page models) | 0.744 / 0.712 | 0.855 / 0.830 | 0.869 / 0.855 |
| S_soup3k (3 extra-page models only) | 0.745 / 0.724 | — | — |
| ZK_hk1k_seed2 ep6 (best single InternVL) | 0.734 / 0.740 | 0.829 / 0.858 | 0.841 / 0.862 |
| SV_lr6e-5 ep6 (best SmolVLM) | 0.545 / 0.589 | 0.653 / 0.713 | 0.704 / 0.732 |

- Averages: neither 13k nor 3k beats soup10 on both sets (gold up ~0.017, val down 0.008–0.020). soup10 stays best.
  The 3 extra-page seeds at 20:00 give 6 such models for a larger extra-page-only average.

### 2026-10-14 — granite-docling transcript test (TR_test, 16 training pages)
- Speed: 242 s for 16 pages at batch 8 (~15 s/page on one GPU). Turkish/French accents kept (é, è, à, ç, ô ... present).
- Failures: 5 of 16 pages. 3 pages returned no text (25 tokens; granite-docling probably tagged the whole page as one picture — not checked),
  1 page ran 6,000 tokens and left 4 characters (looping location tokens), 1 page hit 6,000 tokens with 226 lines of which 44 unique (line loop).
- Decision: transcribe all 2,465 training rows anyway (8 shards, 2 chains of 4 jobs so the daytime limit holds: TR_s0–s7, jobs 1603873–80).
  When building the reading-stage file, drop rows that are empty, hit the token limit, or have a low unique-line fraction; report how many were dropped.
- Next: the SmolVLM reading stage (Pix2Struct ablation: removing the reading warm-up lowered DocVQA 67.8 → 56.2) on the kept transcripts, then JSON training from it.

### 2026-10-14 — teacher-vs-teacher ceiling on the content score
Second GLM run (`runs/label_ceiling/glm_rerun.jsonl`) scored against reviewed gold, 50 dev gold pages (strict/lenient reproduce the H0 numbers, so the setup matches):
| | strict | lenient | content |
|---|---|---|---|
| raw | 0.899 | 0.912 | 0.967 |
| canonical | 0.912 | 0.925 | 0.967 |
Reading: the teacher disagrees with itself on 9 % of fields but only 3 % of words. Best students now score content 0.84–0.87 on gold
(InternVL average soup10 0.839 raw / 0.865 repaired), so about 0.10 of content F1 is real reading/omission error by the student,
not label noise — this is the part the reading stage and section-wise generation target.

### 2026-10-14 — SmolVLM: 10 epochs at lr 6e-5 (SV_lr6e-5_ep10)
- Hypothesis: SV_lr6e-5 still rose at its last epoch (6), so it was under-trained; 10 epochs should help.
- Lenient F1 gold / val per epoch: ep6 0.583/0.591, ep7 0.566/0.572, ep8 0.617/0.607, ep9 0.596/0.627, ep10 0.623/0.616.
  Validation picks ep9: **gold 0.596 / val 0.627** (content 0.768/0.750); mean over ep6–10 0.597/0.603.
- Against SV_lr6e-5 (0.545/0.589): +0.05 gold, +0.04 val, both above the ~0.02 spread → **adopted as the SmolVLM best**.
- Note the 6-epoch run at ep6 scored 0.545/0.589 while this run at ep6 scores 0.583/0.591: the lr schedule (cosine over 10 epochs
  keeps lr higher at ep6) differs, so these are not the same model at the same point.
- Next: SV_lr6e-5_ep15 (does it still rise? scores ep9–15), and SV_lr1e-4_ep10 (20:00). The queued 6-epoch SV_lr1e-4 and
  SV_lr3e-5_tiles30 were cancelled: they compared against the old 6-epoch baseline. SV_lr6e-5/model deleted (superseded).

### 2026-10-14 — granite-docling-258M first run (GD_lr6e-5): starts the third track
- Setting copied from the SmolVLM recipe (same Idefics3 family): full FT + vision, lr 6e-5, 6 epochs, 12 tiles, augment 0.5, seed 42.
- Lenient gold / val: ep3 0.449/0.474, ep4 0.474/0.519, ep5 0.542/0.546, ep6 0.534/0.539. Validation picks ep5:
  **gold 0.542 / val 0.546** (content 0.676 / 0.690). Mean ep3–6 0.500/0.520.
- Why recall is low (val P 0.58, R 0.50): 11 of 79 val answers hit the 3,000-token limit (looping; granite answers are ~1,070 tokens,
  so the limit is not too short) and 14 have schema errors; SmolVLM ep9 has 5 and 6. Loops cut whole pages from the score.
- Next (GD_lr6e-5_ep10, job queued): 10 epochs, the change that gave SmolVLM +0.05/+0.04. If loops stay high, apply the
  repetition fix that worked for InternVL early on to this track too.

### 2026-10-14 — reading-stage plan (hypothesis H-read, SmolVLM first)
- Source: Pix2Struct (arXiv 2210.03347) ablation — removing the reading warm-up dropped DocVQA 67.8 → 56.2. SmolVLM reads page words
  zero-shot at 0.16 (InternVL 0.87), and the content score shows ~0.10 of word-level misses even for InternVL, so the
  failure it targets is reading, not JSON layout.
- Data: `src/reading_file.py` cleans granite-docling transcripts (cut before a line's third copy; drop < 300 chars or < 80 % unique lines).
  On the first 2 of 8 shards: 598 → 494 kept (357 whole, 137 cut, 97 too short, 7 repetitive).
- Run plan: step 1 `exp.slurm RD_sv` trains SmolVLM on the reading file only ("Write out all text on this page.", 2 epochs, lr 6e-5,
  no per-epoch checkpoints → no scoring, saves model/); step 2 `SV_read_ep10` = the SV_lr6e-5_ep10 recipe started from RD_sv/model.
  Compare against SV_lr6e-5_ep10 (0.596 / 0.627). One change: the reading step.

### 2026-10-14 ~13:15 — paused by the user
- Paused while all queued jobs waited on the romer account CPU limit. Still queued on TRUBA (not cancelled):
  TR_s2–s7 (transcription), SV_lr6e-5_ep15, GD_lr6e-5_ep10, and from 20:00 SV_lr1e-4_ep10, ZK_hk1k_seed3/4/5.
- Done today and recorded above: SV_lr6e-5_ep10 (new SmolVLM best 0.596/0.627), GD_lr6e-5 (granite first run 0.542/0.546),
  content-score ceiling 0.967, transcript test + `src/reading_file.py`.
- On resume: read finished jobs' scores from logs; build the reading file when all 8 transcript shards exist; open user
  questions: (1) score the 204 test pages? (2) switch to release work (CPU speed/cost table, README, release without hadikp)?
- Portfolio reviews (3 personas): hiring 7/10 each; open source 3–4/10 now, ~6/10 after README + test score + data-free release.

### 2026-10-14 15:15 — speed on a laptop CPU (Apple M2 Max, 12 cores, CPU only, float32)
Same 10 validation pages, `src/evaluate.py --device cpu --batch 1 --max-patches 12`, one model at a time; times from evaluate.py's
per-page seconds (generation only), peak RAM from `/usr/bin/time -l`. Model load ≈ 5 s each.
| Model | median s/page (range) | output tokens/s | peak RAM | valid JSON |
|---|---|---|---|---|
| granite-docling-258M (GD_lr6e-5 ep5) | 30 (19–45) | 34.7 | 2.9 GB | 6/10 (2 schema errors, 2 unparsed) |
| SmolVLM-500M (SV_lr6e-5_ep10 ep9) | 59 (36–77) | 18.3 | 3.9 GB | 10/10 |
| InternVL3-1B (S_soup10_vision) | 62 (35–80) | 15.6 | 4.9 GB | 9/10 (1 schema error) |
Reading: all three run on a laptop CPU at 0.5–1 min per page in under 5 GB; the teacher (GLM-5.3-Flash, ~300B, user) needs
data-center GPUs. Colab's 2-core CPU will be slower; `notebooks/colab_speed_test.ipynb` measures it (models on Drive `cv_parser_colab/speed/`).
Not yet measured: bf16 / quantised CPU inference, batching, a GPU figure on the same 10 pages.

## 2026-10-14 ~16:30 — granite-docling: 10 epochs adopted (GD_lr6e-5_ep10)
- One change vs GD_lr6e-5 (6 → 10 epochs). Val-chosen ep9: lenient gold 0.574 / val 0.581 vs 0.542 / 0.546:
  +0.032 / +0.035, above spread on both sets (single seed; granite's own seed spread is not measured yet).
- Val flat from ep8 (0.576–0.581) → more epochs are unlikely to help. Unparsed val answers 11 → 5; schema errors 14 → 16.
- Next (queued, job 1604826): GD_lr1e-4_ep10 — lr 1e-4, one change vs GD_lr6e-5_ep10. Content and repaired scores for
  ep9 not computed yet.

## 2026-10-14 ~16:45 — Disk cleanup (owner approved)
- Deleted weights: Z4–Z8 (default-recipe seeds not in S_soup10_vision; adding Z4–Z6 gave no gain), S_soup13k and
  S_soup3k (averages that did not beat S_soup10_vision on val; rebuildable from members in ~10 min), GD_lr6e-5 (superseded).
- Kept: the 10 S_soup10_vision members (Z ep6, Z1, Z2 ep4+ep6, Z3, ZH, ZH1, ZH2, ZX ep5+ep6), S_soup10_vision,
  ZK_hk1k seed42/1/2 (members for the extra-page average with seeds 3–5 tonight), SV_lr6e-5_ep10, GD_lr6e-5_ep10.
  Eval folders and logs kept for all runs.

## 2026-10-14 ~19:00 — SmolVLM: 15 epochs not adopted (SV_lr6e-5_ep15)
- One change vs SV_lr6e-5_ep10 (10 → 15 epochs). Val-chosen ep11: lenient 0.600 / 0.623 vs 0.596 / 0.627 — within spread.
  Val stays 0.593–0.623 from ep9 to ep15: SmolVLM's score has levelled off at this learning rate and data size.
- SmolVLM best stays SV_lr6e-5_ep10. Next for this track: SV_lr1e-4_ep10 (20:00), then the reading-stage run once
  all 8 transcript shards exist (TR_s4/s5 running, s6/s7 after them).

## 2026-10-14 ~19:15 — SmolVLM: less page area per image token (owner request) — 20 and 30 tiles on the current recipe
- **Mechanism (checked in the model's config):** SmolVLM cuts the resized page into 512 px tiles and gives each tile
  64 tokens (SigLIP patch 16 → 32×32 patches, pixel shuffle scale_factor 4 → 8×8), plus one thumbnail. Tokens per
  tile are fixed, so a larger resize (longest_edge 2048 → 2560 / 3072 px = 12 → 20 / 30 tiles on A4) means each token
  covers less page area: 832 → 1,344 / 1,984 image tokens per page. InternVL gives 256 tokens per 448 px tile.
- **Earlier evidence:** SV_tiles30 at the old lr 1e-5: gold +0.06, val −0.004 (val-chosen); mean over epochs +0.03 / +0.02.
  The retest on a better recipe was cancelled when the baseline changed.
- **Runs (20:00):** SV_lr6e-5_tiles20 (1604889), SV_lr6e-5_tiles30 (1604890); one change each vs SV_lr6e-5_ep10
  (0.596 / 0.627). Adopt if > +0.02 on both sets. Cost if adopted: ~1.6× / 2.4× image tokens → slower CPU prefill.
- **Not tried, and why:** lowering pixel shuffle 4 → 2 (256 tokens per tile, same area) changes the input size of the
  pretrained connector layer, so it would start untrained. Cropping white page margins before tiling also gives
  less area per token at no token cost; margin size on our pages not measured yet.

## 2026-10-14 ~20:05 — granite-docling lr 1e-4 (GD_lr1e-4_ep10): gain on gold, below threshold on val — seed repeats queued
- Val-chosen ep10: lenient 0.609 / 0.596 vs GD_lr6e-5_ep10 0.574 / 0.581: +0.035 gold, +0.015 val (threshold 0.02 on both: not met).
  Mean over ep6–10: +0.007 / +0.008. Val still rising at ep10 (0.592 → 0.596).
- Granite's own seed spread is unknown (InternVL's is ~0.012). Queued seed 1 of both learning rates
  (GD_lr1e-4_ep10_seed1, GD_lr6e-5_ep10_seed1) to measure it and decide on 2-seed means. Best stays GD_lr6e-5_ep10 until then;
  GD_lr1e-4_ep10 weights kept for the comparison.

## 2026-10-14 ~20:25 — SmolVLM reading stage submitted (H-read)
- Transcripts done (8 shards, 2,386 pages). `src/reading_file.py` → `data/processed/reading_train.jsonl`: 1,976 kept
  (1,454 whole, 522 cut at a line's third copy), 410 dropped (367 under 300 characters, 43 under 80 % unique lines).
- Checked: 0 rows from gold pages (by image and by cv_id), 0 from the validation split; 1,161 rows are pages of the
  JSON training file, 815 are other Roboflow pages (more images seen, no JSON labels needed).
- Jobs: RD_sv (1604969) = 2 epochs of "Write out all text on this page." on that file, lr 6e-5, saves model/ only;
  SV_read_ep10 (1604970, starts after RD_sv succeeds) = SV_lr6e-5_ep10 recipe from RD_sv/model.
  Compare with SV_lr6e-5_ep10 (0.596 / 0.627); adopt if > +0.02 on both sets.

## 2026-10-14 ~20:50 — SmolVLM lr 1e-4 (SV_lr1e-4_ep10): not adopted
- Val-chosen ep9: lenient 0.612 / 0.624 vs SV_lr6e-5_ep10 0.596 / 0.627 (+0.016 / −0.003), within spread.
  Together with 15 epochs (no gain), learning rate and epoch count are exhausted for SmolVLM at this data size.
  Remaining SmolVLM levers queued: more tiles per page (SV_lr6e-5_tiles20/30) and the reading stage (RD_sv → SV_read_ep10).

## 2026-10-14 ~22:10 — InternVL extra-page seeds 3–5 done; 6-seed and 16-model averages submitted
- Val-chosen lenient gold / val: seed3 0.703 / 0.712, seed4 0.678 / 0.715, seed5 0.729 / 0.724. With seeds 42/1/2 the
  6-seed mean is the H17 test (adopt the extra pages if the mean beats the default recipe's 0.705 / 0.708 by > 0.02 on both).
- S_soup6k (6 ZK seeds; the 3-seed average was 0.745 / 0.724) and S_soup16k (10 S_soup10_vision members + 6 ZK) submitted.
  Compare with S_soup10_vision 0.728 / 0.732; choose on val.
- **6-seed mean (extra pages): 0.716 / 0.721** (SD 0.021 / 0.010) vs the default recipe's 9-seed mean 0.705 / 0.708:
  +0.011 / +0.013. The 3-seed result (0.729 / 0.725) that led to adopting the extra pages was the high end; on 6 seeds
  the gain is below the 0.02 threshold on both sets. Status changed: extra pages = small, uncertain gain, not a proven one.
  (They also cannot go into a public release: hadikp licence unknown.)

## 2026-10-14 ~23:00 — granite-docling: lr 1e-4 adopted on 2-seed means
| lr | seed 42 | seed 1 | mean (gold / val) |
|---|---|---|---|
| 6e-5 | 0.574 / 0.581 | 0.610 / 0.589 | 0.592 / 0.585 |
| 1e-4 | 0.609 / 0.596 | 0.622 / 0.635 | **0.616 / 0.616** |
- +0.024 / +0.031 on 2-seed means, above 0.02 on both sets → lr 1e-4 adopted. Granite seed spread: 0.013–0.036 between
  two seeds of the same recipe, at least as large as InternVL's (~0.012 SD); single granite runs need repeats.
- Next: weight average of the two lr 1e-4 seeds (GD_soup2_lr1e-4; model soups, Wortsman et al. 2022, +0.024 on InternVL)
  and seeds 2 and 3 (a 4-seed mean and a 4-member average). Weights of the lr 6e-5 runs deleted.

## 2026-10-14 ~23:15 — SmolVLM 20 tiles (less page area per image token): likely gain, seed repeats queued
- SV_lr6e-5_tiles20 vs SV_lr6e-5_ep10 (12 tiles): val-chosen +0.038 gold / +0.018 val (val just under 0.02);
  mean over ep6–10 +0.029 / +0.036; every 20-tile epoch ≥ 0.614 gold and ≥ 0.627 val, above most baseline epochs.
- Seed 1 of both tile counts queued (as done for granite) to decide on 2-seed means. SV_lr6e-5_tiles30 still running.

## 2026-10-14 ~23:40 — SmolVLM 30 tiles ADOPTED (owner's idea: less page area per image token)
| tiles (image tokens / page) | val-chosen gold / val | mean ep6–10 |
|---|---|---|
| 12 (832) | 0.596 / 0.627 | 0.597 / 0.603 |
| 20 (1,344) | 0.634 / 0.645 | 0.626 / 0.639 |
| 30 (1,984) | **0.645 / 0.656** | **0.644 / 0.636** |
- 30 tiles beats 12 by +0.049 / +0.029 (val-chosen) and +0.047 / +0.033 (epoch means): above 0.02 on both sets → adopted.
  Gains rise with tile count (12 → 20 → 30) on both measures: SmolVLM's 64 tokens per 512 px tile were too coarse for CV text.
- Largest single-step gain on the SmolVLM track since the learning-rate change. Val still rising at ep10 (0.639 → 0.656).
- Cost: 2.4× image tokens → slower CPU prefill (not yet measured; 12 tiles was 59 s/page median on the M2 Max).
- Queued: SV_lr6e-5_tiles30_seed1 (seed check). Next ideas: more than 30 tiles (code allows 12/20/30 only; 4096 px ≈ 48 tiles),
  the reading stage on 30 tiles if SV_read_ep10 helps, and margin cropping.
- **S_soup6k result (23:50):** average of the 6 extra-page seeds: 0.721 / 0.730 vs S_soup10_vision 0.728 / 0.732 — no gain
  (the 3-seed average was 0.745 / 0.724). Not adopted; its weights deleted. S_soup16k still scoring.
- **S_soup16k result (00:00):** S_soup10_vision members + 6 extra-page seeds: 0.743 / 0.731 vs 0.728 / 0.732 (+0.015 gold,
  −0.001 val). Chosen on val → S_soup10_vision stays the InternVL best. Weights deleted. Extra hadikp pages closed for InternVL:
  no proven gain as single seeds (6-seed mean), as a 6-model average, or added to the best average.

## 2026-10-15 ~00:05 — tile count carried over: granite-docling 30/48 tiles, SmolVLM 48 tiles
- granite-docling uses the same image processor as SmolVLM (checked in its config: 512 px tiles, 64 tokens each,
  longest_edge 2048 by default), so the SmolVLM 12 → 30 tile gain (+0.049 / +0.029) may transfer.
- `src/train_lora.py`: `--max-patches 48` added (longest_edge 4096 px). Checked with the SmolVLM processor on a white
  1240×1754 (A4) image: 2048 → 13 images / 832 tokens, 3072 → 31 / 1,984, 4096 → 49 / 3,136.
- Runs: SV_lr6e-5_tiles48 (1605113, one change vs SV_lr6e-5_tiles30); GD_lr1e-4_tiles30_seed42 and _seed1 (1605114/5,
  vs GD_lr1e-4_ep10 2-seed mean 0.616 / 0.616); GD_lr1e-4_tiles48_seed42 (1605116). 48 tiles may run out of GPU memory
  (no gradient checkpointing); the job then fails and the log shows it.
- **GD_soup2_lr1e-4 (00:10):** weight average of the two granite lr 1e-4 seeds: 0.643 / 0.634 vs the members' mean
  0.616 / 0.616 (+0.027 / +0.018) and the best member (seed 1) 0.622 / 0.635. Averaging helps granite as it did InternVL;
  val ties the best single seed. Decision waits for the 4-seed average (seeds 2, 3 running). Weights kept.

## 2026-10-15 ~00:40 — SmolVLM reading stage (H-read, Pix2Struct-style warm-up): rejected
- SV_read_ep10 vs SV_lr6e-5_ep10 (both 12 tiles): val-chosen +0.005 gold / −0.023 val; mean over ep6–10 −0.007 / −0.008.
  The reading step lowered the starting JSON loss (0.86 vs 1.17) but not the final score.
- Likely reason: at 12 tiles the image detail, not the reading skill, limits SmolVLM — the tile-count runs gained
  +0.05 from resolution alone. Text the model cannot resolve cannot be learnt by transcription either.
  Other possible reasons, not tested: only 2 reading epochs; transcripts come from granite-docling, not checked by hand.
- Not repeated at 30 tiles for now: the 12-tile result gives no sign of a gain. Weights of RD_sv and SV_read_ep10 deleted.

## 2026-10-15 ~02:00 — SmolVLM tile count, 2-seed means: 30 tiles stays adopted; 20 tiles scores the same as 30
- Seed-1 repeats ended (SV_lr6e-5_tiles12_seed1, _tiles20_seed1, _tiles30_seed1). Lenient F1 gold / val:

| Tiles (image tokens) | Seed 42 val-chosen | Seed 1 val-chosen | 2-seed mean, val-chosen | 2-seed mean of ep6–10 means |
|---|---|---|---|---|
| 12 (832) | 0.596 / 0.627 | ep8 0.550 / 0.617 | 0.573 / 0.622 | 0.584 / 0.605 |
| 20 (1,344) | 0.634 / 0.645 | ep7 0.610 / 0.650 | 0.622 / 0.648 | 0.621 / 0.643 |
| 30 (1,984) | 0.645 / 0.656 | ep6 0.587 / 0.651 | **0.616 / 0.654** | **0.630 / 0.639** |

- 30 vs 12: +0.043 / +0.032 (val-chosen), +0.046 / +0.034 (epoch means): above 0.02 on both sets; adoption holds.
- 20 vs 30: −0.006 / +0.006 and +0.009 / −0.004: no difference. 20 tiles is cheaper (1,344 vs 1,984 image tokens), so
  for CPU speed 20 may be the better product choice; not measured yet.
- Seed spread at 30 tiles is large on gold (0.645 vs 0.587 val-chosen; epoch means 0.644 vs 0.615): single-seed
  SmolVLM results on gold need repeats.
- Next: SV_soup2_tiles30 (1605213) = weight average of the two 30-tile seeds (granite's 2-seed average gave +0.027).
  SV_lr6e-5_tiles48 (48 tiles) still running.
- Weights of the 12- and 20-tile runs deleted (eval folders kept).

## 2026-10-15 ~02:00 — Key-by-key generation (user's idea): code written, smoke test queued
- Hypothesis (user): the model writes one key's whole value per call (quotes/brackets included, `null` when the page
  has none); code writes the keys. 29 keys (17 basics keys + 12 sections). The prefix of each key holds all earlier keys
  with their values. Related published work: Alibaba SmartResume (arXiv 2510.09722) splits extraction into 3 parallel
  sub-tasks (basic info, work, education) and reaches 0.964 F1 on a private set with text input; ExtractBench
  (2602.12247) and LMDX also generate per field / per section. Failure modes it targets: whole-page answers that loop
  or break (one bad value now costs one key, not the page), missed sections (every key is asked).
- Data: the same 1,393 training pages; each page gives 29 samples (55.7 % of values are null). Training draws N random
  keys per page per epoch (`--keywise N`); first real run: N = 8.
- Code: src/keywise.py (new), `--keywise` in src/train_lora.py and src/evaluate.py, tests/test_keywise.py (all 1,726
  labelled pages split and re-assemble with no field lost). Training prompt ids are built exactly as generation builds
  them (prompt through the processor, value ids appended), so no BPE merge across the prefix/value boundary.
- KW_smoke (1605211): InternVL, 40 pages × 2 keys, 1 epoch, then 4-page key-by-key scoring; tests that it runs.

## 2026-10-15 ~02:30 — granite 30 tiles: not adopted (gold drops)
| Run | Val-chosen gold / val | Mean ep6–10 gold / val |
|---|---|---|
| GD_lr1e-4_tiles30_seed42 | ep10 0.536 / 0.614 | 0.568 / 0.589 |
| GD_lr1e-4_tiles30_seed1 | ep10 0.596 / 0.620 | 0.587 / 0.607 |
| 2-seed mean | 0.566 / 0.617 | 0.578 / 0.598 |
- vs 12-tile lr 1e-4 2-seed mean 0.616 / 0.616: −0.050 gold / +0.001 val. More tiles do not help granite, unlike SmolVLM
  (+0.04). Cause not tested; one guess: granite-docling was pre-trained at its default 2048 px edge (12 tiles).
  Weights deleted. GD_lr1e-4_tiles48_seed42 still running.

## 2026-10-15 ~03:00 — SmolVLM 48 tiles: no gain over 30; not adopted
- SV_lr6e-5_tiles48 (seed 42, 3,136 image tokens): val-chosen ep9 0.638 / 0.647; mean ep6–10 0.637 / 0.640.
- vs 30-tile 2-seed means (0.616 / 0.654 val-chosen; 0.630 / 0.639 epoch means): +0.022 / −0.007 and +0.007 / +0.001,
  inside the seed spread (30-tile seeds differ by 0.058 on gold). 48 tiles costs 58 % more image tokens and 29 % more
  training time (2:27 vs 1:54). No seed repeat: the epoch means show no gain to confirm. Weights deleted.
- Tile-count result for SmolVLM: 12 → 20 helps (+0.05 gold), 20 / 30 / 48 score the same.

## 2026-10-15 ~03:35 — granite 48 tiles: not adopted
- GD_lr1e-4_tiles48_seed42: val-chosen ep6 0.594 / 0.601; mean ep6–10 0.583 / 0.591 (vs 12 tiles 0.616 / 0.616).
- Granite tile count closed: 30 and 48 tiles both score below 12. Weights deleted. (The monitor missed this job's end
  during a re-arm gap; scores read from rescore.json directly.)

## 2026-10-15 ~03:50 — key-by-key runs queued on all three tracks (start after KW_smoke succeeds)
- KW_smoke training passed: "keywise check ok: learnt '["ghofran.challouf@gmail.com"]<|im_end|>'" (value + end token only).
- Each run: the track's current recipe, one change: `--keywise 8 --epochs 3` (24 samples per page = ~4× the
  baseline's training compute; each key seen ~0.8× per page), scored key-by-key from epoch 2 with 2,000 tokens per value.
  KW_ivl_k8 (1605219) vs InternVL 9-seed mean 0.705 / 0.708; SV_kw_k8 (1605220, 30 tiles) vs SmolVLM 30-tile 2-seed
  mean 0.616 / 0.654; GD_kw_k8 (1605221) vs granite 12-tile 2-seed mean 0.616 / 0.616. --time 06:00 each.
- Not yet checked: SmolVLM and granite mask checks (they run at job start and stop the job if wrong); scoring time per page.

## 2026-10-15 ~04:00 — granite 4-seed weight average: new granite best 0.660 / 0.649
- GD_soup4_lr1e-4 (1605209) = average of GD_lr1e-4_ep10 seeds 42/1/2/3 at their val-chosen epochs: 0.660 / 0.649.
- vs the 4 seeds' mean (val-chosen: 0.616 / 0.616 for seeds 42/1; seeds 2 ep8 0.621 / 0.601, 3 ep9 0.601 / 0.622)
  and the 2-seed average GD_soup2_lr1e-4 0.643 / 0.634: more members, higher score, as for InternVL (S_soup10).

## 2026-10-15 ~04:20 — SmolVLM 2-seed average; key-by-key debug; more seeds queued
- SV_soup2_tiles30 (1605213): 0.644 / 0.660 vs the 2 seeds' mean 0.616 / 0.654 (+0.028 / +0.006). SmolVLM best on val.
- KW_smoke_eval (4 gold pages, the 5-step smoke model, invalid values saved): the model reads the right text after the
  prefix (e.g. basics.name → "Alaeddine Dammak") but, barely trained, keeps writing whole-page JSON after the value
  ('"Elyes Tarmiz","age":"25",…'), so 2 values per page are invalid. Prompt building and value collection work; whether
  the model learns to stop after one value is what KW_*_k8 test. All three mask checks passed (InternVL 'null<|im_end|>',
  SmolVLM 'null<end_of_utterance>', granite '["(+216) 51 822 084"]<|end_of_text|>').
- Queued for larger weight averages (night): GD_lr1e-4_ep10_seed4/5 (1605223/24), SV_lr6e-5_tiles30_seed2 (1605225).

## 2026-10-15 ~06:30 — granite 6-seed weight average: not better than 4 seeds on val
- GD_soup6_lr1e-4 (1605347; seeds 42/1/2/3/4/5 at val-chosen epochs): 0.675 / 0.629 vs GD_soup4_lr1e-4 0.660 / 0.649.
  +0.015 gold / −0.020 val. Chosen on val, the 4-seed average stays the granite best. Seeds 4 (0.654 / 0.608) and 5
  (0.567 / 0.610) are weaker on val than seeds 1–3; a val-greedy member choice could be tried, but gains of this size are
  inside the spread.

## 2026-10-15 ~05:15 — key-by-key, first score: InternVL epoch 2 well below whole-page training
- KW_ivl_k8 epoch 2, gold 50 (raw strict field F1): 0.570 (P 0.596 / R 0.545) vs S_soup10_vision 0.713 (P 0.739 / R 0.688).
  40 valid, 10 schema errors (literal values outside the schema in projects/skills/work, extra keys in certificates);
  6 values not valid JSON, 5 values hit the 2,000-token limit. Epoch 3 and val still scoring.
- The model predicts about as many fields as the whole-page model, but more of them are wrong: misread names
  ("Wevio" for "Wevioo", "Cartridge" for "Carte"), text placed under other fields. Recall drops in every section
  (work 0.53 vs 0.70, skills 0.51 vs 0.71, projects 0.44 vs 0.63).
- Likely cause (not tested): far fewer gradient updates on answer text. Whole-page training learns every label token 6
  times (6 epochs); with 8 random keys of 29 per page per epoch, by epoch 2 each key's value was learnt ~0.55 times per
  page — about 10× fewer answer-token updates for the long sections.
- Next test (queued, 1605382): KW_ivl_soupinit = start from S_soup10_vision (already reads well) and train key-by-key
  for 1 epoch (8 keys per page). If the drop is from too little training on the text, this run should score near the
  soup; if the key-by-key format itself hurts, it will drop too.
- Code: `load_processor` moved into train_lora.py (shared with evaluate.py), so training can start from weights with no
  processor files (weight averages, checkpoints).

## 2026-10-15 ~05:40 — SmolVLM 3-seed weight average: new SmolVLM best 0.644 / 0.676
- SV_soup3_tiles30 (1605370; 30-tile seeds 42 ep10, 1 ep6, 2 ep8): 0.644 / 0.676 vs SV_soup2_tiles30 0.644 / 0.660
  (val +0.016) and the 3-seed mean 0.619 / 0.654 (+0.025 / +0.022).

## 2026-10-15 ~06:00 — key-by-key InternVL from the base model: below whole-page, still rising at the last epoch
| KW_ivl_k8 | Lenient gold / val | Content gold / val |
|---|---|---|
| epoch 2 | 0.590 / 0.596 | — |
| epoch 3 (val-chosen) | 0.647 / 0.650 | 0.809 / 0.808 |
| whole-page, 9-seed mean | 0.705 / 0.708 | — |
| S_soup10_vision | 0.728 / 0.732 | 0.839 / 0.852 |
- −0.058 / −0.058 vs the whole-page seed mean, but +0.057 / +0.054 from epoch 2 to 3: the run had not finished
  learning, consistent with the too-few-updates explanation. Epoch 3: 0 unparsed pages, 7 / 13 schema errors (values
  outside the schema's allowed lists, extra keys), 4 / 3 invalid values, 3 / 2 values at the token limit.
- Content F1 gap (−0.03 / −0.04 vs the soup) is smaller than the lenient gap: more of the loss is wording / field
  placement than missed text.
- Weights deleted. Waiting for KW_ivl_soupinit (start from the soup) to separate "too little training" from "format".
- Queued KW_ivl_k8_ep6: same as KW_ivl_k8 but 6 epochs (48 samples per page, each key's value learnt ~1.7× per page),
  scored at epochs 5 and 6. Tests directly whether the gap to whole-page training closes with more training.

## 2026-10-15 ~06:10 — key-by-key from the best InternVL weights: drops, so the format costs accuracy (after 1 epoch)
| | Lenient gold / val | Content gold / val |
|---|---|---|
| S_soup10_vision (start weights, whole-page) | 0.728 / 0.732 | 0.839 / 0.852 |
| KW_ivl_soupinit, 1 key-by-key epoch | 0.681 / 0.667 | 0.818 / 0.795 |
- −0.047 / −0.065 lenient. A model that already reads the pages well loses accuracy when retrained to the key-by-key
  format, so the gap in KW_ivl_k8 is not only too little training. Uncertain: one epoch may be a transition state
  (old format unlearnt, new one not yet learnt); only 1 epoch was scored.
- Schema errors 9 / 14 pages (S_soup10_vision has 8 / 10; these pages are still scored). Invalid values 6 / 6, values at
  the token limit 4 / 3.
- Weights deleted. KW_ivl_k8_ep6 (6 epochs from the base model) still decides whether more training closes the gap; if
  it stays below ~0.70 the key-by-key format is dropped for InternVL.
- Queued KW_ivl_soupinit_ep3 (1605482): same start weights, 3 key-by-key epochs, each scored. Tests the open question
  above: if epochs 2–3 recover to ~0.73 the 1-epoch drop was a transition state; if they stay near 0.68 the format costs accuracy.

## 2026-10-15 ~06:20 — key-by-key granite: far below whole-page, still rising
- GD_kw_k8 (1605221; 3 epochs, lr 1e-4, 12 tiles, 8 keys per page per epoch): ep2 0.316 / 0.342, ep3 (val-chosen)
  0.474 / 0.463 lenient, content 0.640 / 0.615. Granite whole-page 12-tile mean (10 epochs): 0.616 / 0.616.
- +0.16 / +0.12 from epoch 2 to 3: not finished learning (3 epochs × 8 keys = each key ~0.8 times per page; the
  whole-page baseline trains 10 epochs). Granite repeats text often: 22 of 79×29 val values stopped at the token limit,
  27 invalid values (8 / 11 on gold).
- Not adopted. Weights deleted. A longer granite key-by-key run waits for the InternVL answer (KW_ivl_k8_ep6 and
  KW_ivl_soupinit_ep3): if InternVL shows the format costs accuracy even when trained longer, granite is not retried.

## 2026-10-15 ~08:10 — key-by-key SmolVLM: below whole-page, still rising
- SV_kw_k8 (1605220; 3 epochs, lr 6e-5, 30 tiles, 8 keys per page per epoch): ep2 0.446 / 0.470, ep3 (val-chosen)
  0.530 / 0.526 lenient, content 0.715 / 0.699. SmolVLM whole-page 30-tile 3-seed mean (10 epochs): 0.619 / 0.654.
- +0.08 / +0.06 from epoch 2 to 3: not finished learning, as for granite. Invalid values 12 / 18, values at the token
  limit 5 / 15, schema errors 5 / 11 pages.
- Not adopted. Weights deleted. Same decision rule as granite: retried with more epochs only if the InternVL runs show the
  format does not cost accuracy when trained longer.

## 2026-10-15 ~08:05 — hypothesis (SmolVLM and granite): NEFTune embedding noise reduces overfitting on 1.4K pages
- Source: Jain et al. 2023, "NEFTune: Noisy Embeddings Improve Instruction Finetuning" (arXiv 2310.05914): uniform
  noise of size alpha/sqrt(L·d) added to the input embeddings during training only; reported gains in output quality
  and less overfitting to the small fine-tuning set. Built into transformers (`neftune_noise_alpha`).
- Failure mode targeted: both small models fit the training pages well but score lower on unseen pages, and their
  scores vary strongly between seeds and epochs (seed spread up to 0.058 on gold), a sign of overfitting.
- Code: `--neftune ALPHA` in train_lora.py. Checked on TRUBA: noise is added in training mode only (max 0.0038 granite,
  0.0029 SmolVLM at alpha 5, L = 3,000), against an embedding std of 0.13 / 0.11. L counts the image placeholder tokens
  too, so the noise is weaker relative to the embeddings than in the paper (~3 % of std at alpha 5); chose alpha 15, the
  top of the paper's range.
- Queued: SV_neft15_tiles30 (1605665; SmolVLM, 30 tiles, lr 6e-5, 10 epochs, seed 42) vs the 30-tile 3-seed mean
  0.619 / 0.654; GD_neft15 (1605666; granite, 12 tiles, lr 1e-4, 10 epochs, seed 42) vs the 12-tile seed mean
  0.616 / 0.616. One seed each: a gain must exceed the seed spread, so a gain under ~0.03 needs a second seed.

## 2026-10-15 ~09:25 — key-by-key from the best InternVL weights, 3 epochs: gap shrinks to −0.02 / −0.03
| KW_ivl_soupinit_ep3 (1605482) | Lenient gold / val | Content gold / val |
|---|---|---|
| epoch 1 (lr still high) | 0.524 / 0.524 | — |
| epoch 2 | 0.666 / 0.653 | — |
| epoch 3 (val-chosen) | 0.711 / 0.704 | 0.842 / 0.840 |
| start weights S_soup10_vision (whole-page) | 0.728 / 0.732 | 0.839 / 0.852 |
- The 1-epoch run's drop (0.681 / 0.667) was partly a transition state: with 3 epochs the gap is −0.017 / −0.028
  lenient, and content F1 is equal on gold (+0.003) and −0.012 on val. The model reads the same text; the remaining
  loss is wording / field placement. Still rising (+0.045 / +0.051 from epoch 2 to 3).
- Errors are now rare: invalid values 2 / 3, values at the token limit 1 / 1, schema errors 7 / 7 pages (whole-page 8 / 10).
- Not adopted yet (below the start weights). Cost to keep in mind: 29 generation calls per page instead of 1.
- Weights deleted. Queued KW_ivl_soupinit_ep6 (1605813): same start, 6 epochs, scored at epochs 5–6. If it passes
  0.728 / 0.732 by more than 0.02 on both sets, key-by-key is adopted for InternVL.

## 2026-10-15 ~09:50 — NEFTune granite, seed 42: small gain over the same seed, within spread of the seed mean
| granite, lr 1e-4, 12 tiles, 10 epochs | val-chosen (gold / val) | mean ep6–10 (gold / val) |
|---|---|---|
| GD_neft15 (alpha 15, seed 42) | ep9 0.632 / 0.623 | 0.616 / 0.596 |
| GD_lr1e-4_ep10 (same seed, no noise) | ep10 0.609 / 0.596 | 0.587 / 0.582 |
| no-noise mean of 6 seeds (val-chosen) | 0.616 / 0.616 | — |
- Same seed: +0.023 / +0.027 val-chosen, +0.029 / +0.014 epoch mean. Against the 6-seed mean only +0.016 / +0.007, and
  granite seeds spread up to 0.058 on gold, so one seed does not decide. Queued GD_neft15_seed1 (seed 1; its no-noise
  run scored 0.622 / 0.635). GD_neft15 weights kept for a possible noise-trained weight average.

## 2026-10-15 ~10:10 — NEFTune SmolVLM, seed 42: no gain
| SmolVLM, lr 6e-5, 30 tiles, 10 epochs, seed 42 | val-chosen (gold / val) | mean ep6–10 (gold / val) |
|---|---|---|
| SV_neft15_tiles30 (alpha 15) | ep10 0.642 / 0.651 | 0.629 / 0.639 |
| SV_lr6e-5_tiles30 (same seed, no noise) | ep10 0.645 / 0.656 | 0.644 / 0.636 |
- −0.003 / −0.005 val-chosen, −0.015 / +0.003 epoch mean: no effect for SmolVLM. Not repeated; weights deleted.
  SmolVLM needs a new published-work hypothesis (in preparation).
- Queued GD_neft15_seed2 (seed 2; its no-noise run scored 0.621 / 0.601) next to seed 1, so the granite NEFTune
  decision rests on 3 paired seeds.
- Note for the next SmolVLM hypothesis: token-limit hits are rare for all three track bests (0–4 pages per set), so loops
  are not the main loss. Lenient → content gap is 0.17 / 0.12 for SV_soup3 and 0.14 / 0.13 for GD_soup4 (InternVL 0.11 /
  0.12): most lost points are correct text in the wrong field or split differently, not misread text.

## 2026-10-15 ~10:40 — hypothesis (SmolVLM): 993 extra pages fix field-placement errors with the current recipe
- Target: SmolVLM's lenient → content gap (0.17 / 0.12) = correct text in the wrong field or split differently. A
  smaller model needs more examples of the structure; the only earlier SmolVLM test with the extra hadikp pages
  (SV_hk1k_lr3e-5, 2026-10-14) used the old recipe (lr 3e-5, 12 tiles, 6 epochs) and is not comparable.
- Source: not a specific paper claim (not re-checked): the general finding that fine-tuned accuracy rises with the
  amount of task data, and our own InternVL result: for InternVL the extra pages gave +0.011 / +0.013 (ZK 6-seed mean vs 9-seed mean, below 0.02).
- Queued for 20:00: SV_hk1k_tiles30 (current SmolVLM recipe: lr 6e-5, 30 tiles, 10 epochs, seed 42,
  labels_train_v6_hk1k.jsonl, 2,465 pages) vs SV_lr6e-5_tiles30 seed 42 (0.645 / 0.656) and the 3-seed mean 0.619 / 0.654.

## 2026-10-15 ~11:15 — key-by-key InternVL from the base model, 6 epochs: equal to whole-page on gold, below on val
| KW_ivl_k8_ep6 (1605424) | Lenient gold / val | Content gold / val |
|---|---|---|
| epoch 5 | 0.679 / 0.673 | — |
| epoch 6 (val-chosen) | 0.707 / 0.685 | 0.846 / 0.821 |
| KW_ivl_k8 (3 epochs), epoch 3 | 0.647 / 0.650 | 0.809 / 0.808 |
| whole-page, 9-seed mean | 0.705 / 0.708 | — |
- Doubling training closed most of the gap: +0.060 / +0.035 over 3 epochs; now +0.002 / −0.023 vs the whole-page seed
  mean (one seed; within spread on gold, slightly beyond on val). Still rising at epoch 6 (+0.028 / +0.012).
- Errors rare: invalid values 3 / 4, values at the token limit 3 / 2, schema errors 8 / 6 pages.
- Conclusion so far: key-by-key matches whole-page accuracy when trained enough, but has not beaten it, and costs 29
  generation calls per page. KW_ivl_soupinit_ep6 (best weights + 6 key-by-key epochs) decides adoption. Weights deleted.
- Moved SV_hk1k_tiles30 from 20:00 to now (1605899; a daytime slot was free).

## 2026-10-15 ~11:50 — NEFTune granite, seed 1: below the same seed without noise
- GD_neft15_seed1 (1605833): ep6 0.576 / 0.562, ep7 0.602 / 0.579, ep8 0.619 / 0.608, ep9 0.613 / 0.610, ep10 0.618 / 0.600.
  Val-chosen ep9 0.613 / 0.610 vs 0.622 / 0.635 without noise (−0.009 / −0.025). Seed 42 was +0.023 / +0.027, so the
  2-seed paired mean is +0.007 / +0.001: no effect so far. Seed 2 (running) completes the 3-seed check. Weights deleted.

## 2026-10-15 ~12:50 — NEFTune granite, 3 paired seeds: no effect, rejected
| seed | with noise (val-chosen) | without noise | difference |
|---|---|---|---|
| 42 | ep9 0.632 / 0.623 | 0.609 / 0.596 | +0.023 / +0.027 |
| 1 | ep9 0.613 / 0.610 | 0.622 / 0.635 | −0.009 / −0.025 |
| 2 | ep7 0.616 / 0.620 | 0.621 / 0.601 | −0.005 / +0.019 |
| mean | 0.620 / 0.618 | 0.617 / 0.611 | +0.003 / +0.007 |
- Below the 0.02 bar on both sets; rejected for granite (and SmolVLM, one seed −0.003 / −0.005). Weights of all three
  noise runs deleted. Seed 42's +0.02 was seed noise: granite single-seed differences of ±0.025 occur both ways.

## 2026-10-15 ~12:50 — where the best InternVL model loses values (S_soup10_vision, lenient)
| lost gold value | gold (4,612 values) | val (7,203) |
|---|---|---|
| missing, nothing similar written | 10.6 % | 12.3 % |
| page not parsed (3 gold pages) | 6.7 % | 0 % |
| joined with other text into one value | 4.6 % | 6.8 % |
| written only in part (split) | 3.1 % | 2.9 % |
| misread (≥ 80 % similar text in the section) | 2.9 % | 3.7 % |
| same text under another field | 1.8 % | 2.1 % |
| date with added digits ("2021-04" → "2021-04-21") | 0.1 % | 0.0 % |
- Joined + split + other field ≈ 9.5 % / 11.8 % = the lenient → content gap. About half of it matches the teacher's own
  run-to-run disagreement (lenient 0.912 vs content 0.967, H0). Missing values + unparsed pages cost more than the gap.
- Script: scratchpad gap.py (one-off, not kept).

## 2026-10-15 ~13:20 — SmolVLM with 993 extra pages (SV_hk1k_tiles30): no gain, rejected
- Ran: SmolVLM-500M, 30 tiles, lr 6e-5, 10 epochs, seed 42, `labels_train_v6_hk1k.jsonl` (2,386 pages instead of 1,393);
  one change vs SV_lr6e-5_tiles30 (same seed).
- Lenient gold / val: ep6 0.636 / 0.633, ep7 0.629 / 0.633, ep8 0.651 / 0.645, ep9 0.626 / 0.631, ep10 0.624 / 0.631.
  Val-chosen ep8 0.651 / 0.645 (content 0.792 / 0.770) vs 0.645 / 0.656 same seed without the pages: +0.006 / −0.011.
  Mean ep6–10 0.633 / 0.635.
- Went right/wrong: same outcome as InternVL (ZK, extra pages: gain below 0.02). With 1.7× the pages, val is lower at every
  epoch. Likely reason (not checked): the extra pages differ in layout/language from the gold and val pages.
- Not repeated on more seeds: one seed shows no sign of a gain, and the InternVL 6-seed result points the same way.
- Next (SmolVLM): seeds 3 and 4 at the current recipe (1605969/70) for a 5-seed weight average (model soups, Wortsman et al.
  2022; on InternVL 10 members beat 4). Published-work hypothesis for SmolVLM and granite still to be chosen.

## 2026-10-15 ~13:35 — hypothesis (granite): a 10× lower learning rate for the vision encoder keeps its pretrained features
- Published work: discriminative fine-tuning, a lower learning rate for lower / pretrained layers (Howard & Ruder 2018,
  ULMFiT, arXiv 1801.06146); mechanism: full fine-tuning distorts pretrained features (Kumar et al. 2022, arXiv 2202.10054).
- Why granite: its vision encoder was trained by IBM on document pages (DocTags) and read page words best of the three
  base models (0.96); at lr 1e-4 the encoder moves at the same rate as the language model.
  Failure mode targeted: misread and missing values (lost-value table, 2026-10-15 ~12:50). Not checked whether
  granite's encoder actually drifts.
- Code: `train_lora.py --vision-lr` (VisionLrTrainer splits Trainer's optimizer groups by vision / rest; toy test:
  the 10× ratio holds through warm-up and cosine decay, weight-decay grouping kept).
- Ran: GD_vlr1e-5 (seed 42, 1605971, now), _seed1 / _seed2 (1605972/3, 20:00). One change vs GD_lr1e-4_ep10
  (lr 1e-4, vision 1e-5). Paired-seed references: 0.609 / 0.596, 0.622 / 0.635, 0.621 / 0.601. Adopt if the 3-seed
  paired mean gain is > 0.02 on both sets.

## 2026-10-15 ~14:50 — key-by-key from the best InternVL weights, 6 epochs: no gain; key-by-key not adopted
- KW_ivl_soupinit_ep6 (1605813): InternVL3-1B, --keywise 8, 6 epochs from S_soup10_vision. 5 h 02 min.
- Lenient gold / val (content): ep5 0.716 / 0.692 (0.850 / 0.825), ep6 0.728 / 0.710 (0.854 / 0.825).
  Val-chosen ep6 0.728 / 0.710 vs start weights 0.728 / 0.732: 0.000 / −0.022. Content +0.015 / −0.027.
- Conclusion over all 5 key-by-key runs: with enough training the format reaches whole-page accuracy on gold but stays
  below it on val, and it needs 29 generation calls per page instead of 1. Not adopted; weights deleted.
- Went right: the format removes broken JSON by construction (each value is checked on its own). Went wrong: the model
  sees only one key at a time while writing, so it cannot use what it already placed elsewhere — a likely reason
  (not checked) for the val drop.

## 2026-10-15 ~15:05 — hypothesis (InternVL): the same 10× lower vision-encoder learning rate
- Published work and mechanism as for granite (~13:35 entry: Howard & Ruder 2018; Kumar et al. 2022).
- Why InternVL: training the vision encoder at full lr helped (+0.011 / +0.022, 3 seeds); a lower encoder lr may keep
  that gain with less drift of the pretrained features. Targets misread and missing values.
- Ran: ZV_vlr1e-6_seed42 / _seed1 / _seed2 (start 20:00), default recipe (lr 1e-5, 6 epochs, image noise 0.5, vision
  trained) with --vision-lr 1e-6. Paired references (val-chosen): seed 42 0.699 / 0.720, seed 1 0.722 / 0.711,
  seed 2 0.697 / 0.693; mean 0.706 / 0.708. Adopt if the paired mean gain is > 0.02 on both sets.

## 2026-10-15 ~15:45 — granite lower vision-encoder lr, seed 42: clearly worse
- GD_vlr1e-5 (1605971): lenient gold / val ep6 0.543 / 0.554, ep7 0.522 / 0.540, ep8 0.547 / 0.577, ep9 0.560 / 0.570,
  ep10 0.519 / 0.576. Val-chosen ep8 0.547 / 0.577 vs 0.609 / 0.596 same seed without it: −0.062 / −0.019;
  mean ep6–10 0.538 / 0.563 vs 0.587 / 0.582 — lower at every epoch.
- Reading: granite's encoder needs to change at the full rate for this task; the features it brings from DocTags
  pretraining are not enough as they are. Seed 1 (running, half done) kept as a check; seed 2 cancelled.
- InternVL runs kept (different encoder, different pretraining), but seed 42 moved to now so its result decides
  whether seeds 1/2 run tonight.

## 2026-10-15 ~16:20 — SmolVLM 5-seed weight average: not better than 3 seeds on val
- SV_soup5_tiles30 (1606302) = average of SmolVLM 30-tile seeds 42/1/2/3/4 at their val-chosen epochs: 0.657 / 0.673
  (lenient gold / val) vs SV_soup3_tiles30 0.644 / 0.676: +0.013 / −0.003. 5-seed mean 0.627 / 0.656.
- Same pattern as granite (6 > 4 seeds on gold, not on val). SV_soup3_tiles30 stays the SmolVLM best (chosen on val).

## 2026-10-15 ~16:45 — InternVL lower vision-encoder lr, seed 42: val drops; not continued
- ZV_vlr1e-6_seed42 (1606259): lenient gold / val ep3 0.645 / 0.630, ep4 0.693 / 0.645, ep5 0.725 / 0.682,
  ep6 0.715 / 0.667. Val-chosen ep5 0.725 / 0.682 vs Z_aug_vision (seed 42) 0.699 / 0.720: +0.026 / −0.038.
- With granite seed 42 also lower (−0.062 / −0.019), the lower encoder rate does not help either model: the
  encoders need to adapt at the full rate. Seeds 1 / 2 cancelled; weights deleted. Granite seed 1 still running.

## 2026-10-15 ~17:00 — lower vision-encoder lr: rejected for both models
- GD_vlr1e-5_seed1: val-chosen ep8 0.597 / 0.597 vs 0.622 / 0.635 same seed without it (−0.025 / −0.038). With seed 42
  (−0.062 / −0.019) and InternVL seed 42 (+0.026 / −0.038), the hypothesis is rejected. Weights deleted.
- Next (milestone): the three best models are scored once on the 204 held-out test pages (truba/test.slurm).

## 2026-10-15 ~17:40 — Milestone: the three best models scored once on the 204 test pages
- truba/test.slurm (gold pages 51–254, never used for a decision). Lenient F1 raw (repaired) / strict / content:
  InternVL S_soup10_vision 0.745 (0.746) / 0.732 / 0.844, unparsed 3; SmolVLM SV_soup3_tiles30 0.677 (0.683) /
  0.667 / 0.788, unparsed 4; granite GD_soup4_lr1e-4 0.633 (0.647) / 0.623 / 0.752, unparsed 16, token limit 12.
- The test scores are at or above the dev scores for InternVL and SmolVLM: choosing on val did not overfit the dev
  pages. SmolVLM is ahead of granite on test, the reverse of gold dev (0.644 vs 0.660): 50 pages cannot separate
  students 0.02 apart. granite's weak point is answers that never close (16 unparsed, 12 at the token limit).

## 2026-10-15 ~17:50 — SmolVLM at 20 tiles: same accuracy, faster on CPU? (queued for 20:00)
- Observation: 20 tiles scored the same as 30 on 2-seed means (2026-10-15 ~02:00), with 1,344 instead of 1,984 image
  tokens per A4 page. On a CPU, prompt processing grows with image tokens, so a 20-tile model would read a page faster.
- Ran: SV_tiles20_seed42 / _seed1 / _seed2 (configs/smolvlm-500m.toml, --max-patches 20), then a 3-seed average to
  compare with SV_soup3_tiles30 (0.644 / 0.676 dev, 0.677 test) and a CPU timing of both.
- Adopt for release if the 20-tile average is within the seed spread of the 30-tile average on both dev sets and
  faster per page on CPU. First jobs submitted with the package layout (PYTHONPATH=src).

## 2026-10-15 17:45 — Token perturbation on its own (ZP_noperturb) — queued
- Why: token perturbation (Nougat, arXiv 2308.13418) entered in lora_v2 together with removing `corrections`, so its
  own effect was never measured. The code swaps one input answer token on ~10 % of pages (~0.11 tokens per page),
  much less than Nougat's per-token rate, so no effect is the expected result.
- Ran: ZP_noperturb_s42 / _s1 / _s2 = configs/internvl3-1b.toml with `--perturb 0` (jobs 1606401–03; seeds 1 and 2
  start 20:00). Baseline: the default recipe (perturb 0.1), 9-seed mean 0.705 / 0.708, paired seeds 42 / 1 / 2.
- Reading: a 3-seed mean more than ≈ 0.014 below on both sets = perturbation helps; within = no measurable effect
  (then the docs say so). Also compare unparsed pages and token-limit hits (perturbation targets loops).

## 2026-10-15 18:10 — Why GRPO (H15) did not raise dev F1: measurements, part 1 (from the saved logs and answers)
Two of the explanations written on 2026-10-13 are contradicted by the GRPO training logs (`log_history.json`):

| run | steps | reward first 50 → last 50 steps | reward std within a group | groups with all-equal rewards | entropy first → last |
|---|---|---|---|---|---|
| lr 1e-6 | 348 | 0.744 → 0.766 | 0.248 → 0.235 | 1 % | 0.075 → 0.078 |
| lr 3e-6 | 348 | 0.768 → 0.795 | 0.225 → 0.210 | 1 % | 0.068 → 0.077 |
| temperature 0.7 | 348 | 0.862 → 0.858 | 0.129 → 0.137 | 1–2 % | 0.022 → 0.023 |
| KL 0.04 | 348 | 0.757 → 0.771 | 0.223 → 0.220 | 0–1 % | 0.067 → 0.078 (KL to start 0.0003) |

- Not true: "the samples of a page score about the same, so there was little to learn from" — the 4 samples of a page
  differ by 0.22–0.25 F1 (std), and only ~1 % of groups give no signal.
- Not true: "the reward rose because sampling became less random" — entropy did not fall at temperature 1.
- The training reward itself rose only +0.02–0.03 in 348 steps.

Dev answers, start model (Z_aug_vision ep6) vs after GRPO, gold pages 1–50, lenient gold values (of 4,612):

| compared with ep6 | identical answers | values gained / lost, pages that stay parsed | values gained / lost, pages that switch parsed ↔ unparsed |
|---|---|---|---|
| supervised ep5 (one epoch earlier) | 8 / 50 | +126 / −106 | +88 / −127 |
| GRPO lr 1e-6 | 10 / 50 | +79 / −75 | +175 / −65 |
| GRPO lr 3e-6 | 4 / 50 | +116 / −147 | +180 / −127 |

- GRPO changed the greedy answers about as much as one more supervised epoch does; on pages that stay parsed, gains
  and losses cancel. The net dev change comes from a few pages switching between parsed and unparsed (each carries
  ~100 values), which is why gold and val moved in opposite directions.
- Part 2 (queued, job P_grpo_probe 1606436, starts 20:00): `experiments/grpo_probe.py` samples 8 answers per page as
  GRPO did (temperature 1) from the start model, on 40 training pages and gold 1–50, and measures where the reward
  spread comes from (broken answers vs content), best-of-8 vs greedy, the share of greedy-missed values found in any
  sample, and how often the teacher-label reward orders two samples differently from the reviewed-label reward.

## 2026-10-15 19:30 — GRPO analysis part 2: first sampling run invalid; corrected teacher-agreement claim
- P_grpo_probe (1606448) sampled with `num_return_sequences=8`. Its samples scored 0.124 mean reward on training
  pages against 0.910 for greedy and ~0.74 in GRPO's own training log: transformers copies `pixel_values` tile by
  tile (repeat_interleave on the tile axis), so each sample saw the page's tiles in the wrong order. Results discarded
  (kept as `runs/P_grpo_probe/samples_scrambled_tiles.jsonl`). The script now repeats the whole input 8 times;
  rerun as P_grpo_probe2 (1606457). Check before reading its results: sample mean reward on training pages near 0.74.
- Docs said the teacher scores 0.912 "on the 204 test pages" / "on the 254 reviewed pages". The 0.912 is a second
  GLM run on the 50 dev gold pages scored against the reviewed labels (H0 above); run 1 vs reviewed is 0.997, and
  review changed 27 of 4,612 lenient values on those pages. README, how-it-works, findings, evaluation, model card
  and the results chart now say this.

## 2026-10-15 20:00 — GRPO analysis part 3: sampling probe (P_grpo_probe2, 1606457, 26 min) — H15 explained
Start model Z_aug_vision checkpoint-528, 8 samples per page (temperature 1, top_p 1, top_k 0, 2,500 tokens), whole
input repeated per sample. Check: sample mean reward on training pages 0.811 vs ~0.74 in GRPO's log at its start
(different page subset) — same range, so the samples are valid.

| set | greedy | mean sample | best of 8 | pages best > greedy + 0.02 | not JSON | reward std in page (JSON only) | greedy-missed values in ≥ 1 sample |
|---|---|---|---|---|---|---|---|
| 40 train pages | 0.910 | 0.811 | 0.925 | 8 / 40 | 13 / 320 | 0.124 (0.070) | 252 / 343 (73.5 %) |
| 50 gold pages | 0.703 | 0.571 | 0.768 | 27 / 50 | 55 / 400 | 0.193 (0.082) | 705 / 1,251 (56.4 %) |

- Lenient vs content order of two JSON samples: opposite in 222 / 915 train pairs (24.3 %) and 232 / 973 gold pairs.
- Teacher-label vs reviewed-label reward order: 0 / 973 opposite — review changed only 27 of 4,612 lenient values on
  these pages, so this cannot show label error; the relevant noise is the teacher's run-to-run variation (H0).
- Verdict on the 2026-10-13 explanations: "no signal" wrong (part 1); "samples became less random" wrong (part 1);
  "the model never samples the missed values" wrong (73.5 % / 56.4 % appear); "little room on training pages"
  supported (greedy 0.910 vs best of 8 0.925); "reward follows teacher splits" supported (24 % order flips).
- Next if GRPO is reopened: unseen pages, keep groups where a sample beats greedy, reward content F1 or recall.

## 2026-10-15 20:15 — Night queue: GRPO on unseen pages; stronger token perturbation for the loop-prone students
**GRPO on pages the supervised model has not seen** (from part 3: on training pages greedy 0.910 vs best of 8 0.925,
on unseen gold pages 0.703 vs 0.768). Start Z_aug_vision checkpoint-528 (0.705 / 0.719), lr 3e-6, 4 samples, pages =
974 hadikp pages with a teacher label (`labels_hadikp_974.jsonl`; never in Z's training; licence unknown, research
only). G4_grpo_unseen: lenient reward (1606603). G5_grpo_unseen_content: content reward, new `--reward content` in
`experiments/train_grpo.py` (1606604; part 3: lenient and content order samples differently in 24 % of pairs).
Compare with G1 lr 3e-6 on training pages (0.718 / 0.704) and the start; adopt only if both dev sets beat the spread.

**Token perturbation at Nougat-like strength** (Nougat, arXiv 2308.13418, aimed at repetition loops). The current
0.1 swaps one token on ~10 % of pages; `--perturb 0.9` swaps ~9 tokens per page on average (geometric count,
mean p / (1 − p)). Tried where loops cost most: granite-docling (12 of 204 test pages hit the token limit)
GD_perturb0.9_s42 / _s1 (1606605–06) vs GD lr 1e-4 seeds 42 / 1 (0.609 / 0.596, 0.622 / 0.635); SmolVLM 30 tiles
SV_perturb0.9_s42 (1606607) vs SV_lr6e-5_tiles30 seed 42 (0.645 / 0.656). Also compare unparsed pages and limit hits.
Runs alongside ZP_noperturb (perturbation off on InternVL), which measures the other direction.
- 20:50 queued GD_perturb0.9_s2 and SV_perturb0.9_s1 (paired seeds; baselines GD lr1e-4 seed 2, SV_lr6e-5_tiles30_seed1).

## 2026-10-15 22:00 — Repository made minimal (log files renamed: RESEARCH_JOURNAL.md -> journal.md, DEVLOG.md -> runs.md)
- main keeps only the adopted recipe. Removed from main, kept in tag archive-2026-10-15: LoRA, key-by-key, canonical labels, NEFTune, vision-lr, constrained decoding, repetition penalty, DPO / GRPO / probe code, data-prep scripts, notebooks, PLAN / DESIGN_NOTES / FINETUNING_RECIPE / DATASET_RESEARCH / literature / results page.
- Checked: training batches identical to the old code (3 pages, seeded noise + perturbation); answers identical on 2 CPU pages; portable job scripts reproduce GD_soup4 0.660 / 0.649.
- 22:25 queued SV_perturb0.9_s2 (third paired seed; baseline SV_lr6e-5_tiles30_seed2).

## 2026-10-15 22:10 — granite-docling perturbation 0.9: first two seeds lower; SmolVLM soup scaling queued
- GD_perturb0.9_s42 (val-chosen ep9): 0.544 / 0.591 vs GD lr1e-4 seed 42 0.609 / 0.596 (−0.065 / −0.005).
- GD_perturb0.9_s1 (val-chosen ep9): 0.615 / 0.605 vs seed 1 0.622 / 0.635 (−0.007 / −0.030).
- Both seeds lower on both dev sets; seed 2 still running. Verdict after seed 2 and the limit-hit counts.
- Found while testing the example notebook on a Mac: train.py trained pure-bf16 weights on Apple MPS (the float32
  upcast ran only on CUDA). Six example pages, 8 epochs: final loss 0.26–0.44 in bf16, 0.02–0.06 in float32. CUDA runs
  already upcast, so no reported result changes; train.py now loads float32 on every device (commit bde403f).
- Queued SV_lr6e-5_tiles30_seed3 / _seed4 (1606944–45) by mistake: the 5-seed SmolVLM average was already tested
  (runs.md: SV_soup5_tiles30 0.657 / 0.673, not adopted). Both jobs found the old log_history.json and ended at once.
- Queued GD_noperturb_s42 / _s1 (perturbation off, one change vs GD lr1e-4 seeds 42 / 1 0.609 / 0.596, 0.622 / 0.635): the
  other direction of the 0.9 runs; on InternVL turning it off made no difference (ZP_noperturb).
- Candidate, not queued: greedy soup (Wortsman et al. 2022, §2: add members by validation score, keep only if val does not
  drop) for InternVL, where uniform 13–19-member averages did not beat 10. Only 4 InternVL member weights remain on disk,
  so it needs retrained members (~21 GB each, ~110 GiB free).
- 22:30 GD_perturb0.9_s2 (val-chosen ep8): 0.591 / 0.629 vs GD lr1e-4 seed 2 0.621 / 0.601 (−0.030 / +0.028).
  3-seed means: perturbation 0.9 0.583 / 0.608 vs 0.1 0.617 / 0.611 (−0.034 / −0.003). **Not adopted** for granite-docling:
  lower on gold, equal on val. Weights of the three runs deleted; eval folders kept.
- Queued GD_noperturb_s2 (third paired seed; baseline seed 2 0.621 / 0.601).
- SV_perturb0.9_s42 (val-chosen ep8): 0.616 / 0.665 vs SV_lr6e-5_tiles30 seed 42 0.645 / 0.656 (−0.029 / +0.009); seeds 1, 2 running.
- SV_perturb0.9_s1 (val-chosen ep8): 0.631 / 0.663 vs SV_lr6e-5_tiles30_seed1 0.587 / 0.651 (+0.044 / +0.012). 2-seed means 0.624 / 0.664 vs 0.616 / 0.654 (+0.008 / +0.010, within spread); seed 2 decides.
- G4_grpo_unseen (lenient reward): 0.708 / 0.714; G5_grpo_unseen_content (content reward): 0.718 / 0.712; start
  Z_aug_vision ckpt-528 0.705 / 0.719, G1 lr 3e-6 on training pages 0.718 / 0.704. Both within spread (val lower): **not
  adopted**. Unseen pages and the content reward did not change the outcome. Weights deleted. Still untested: keep only
  groups where a sample beats greedy. findings.md updated.
- SV_perturb0.9_s2 (val-chosen ep7): 0.636 / 0.668 vs SV_lr6e-5_tiles30_seed2 0.625 / 0.656. 3-seed means 0.628 / 0.665 vs
  0.619 / 0.654 (+0.009 / +0.011): val higher on all three seeds, both within spread. Released SmolVLM is a 3-seed average,
  so queued SV_soup3_perturb0.9 (s42 ckpt-704, s1 ckpt-704, s2 ckpt-616) vs SV_soup3_tiles30 0.644 / 0.676.
- GD_noperturb_s1 (val-chosen ep7): 0.634 / 0.620 vs GD lr1e-4 seed 1 0.622 / 0.635 (+0.012 / −0.015); seeds 42, 2 running.
- GD_noperturb_s42 (val-chosen ep10) 0.618 / 0.629; _s2 (ep7) 0.577 / 0.606; with _s1 3-seed mean 0.610 / 0.618 vs perturb 0.1
  0.617 / 0.611 (−0.007 / +0.007): no effect, as on InternVL (ZP_noperturb). Not adopted.
- SV_soup3_perturb0.9 (1607307, 9 min): 0.674 / 0.658 vs SV_soup3_tiles30 0.644 / 0.676 (+0.030 / −0.018). Not adopted (must
  beat both dev sets). Verdict on token perturbation: no setting beats the spread on any of the three students; recipe keeps
  0.1. Weights of all GD_noperturb, SV_perturb0.9 and SV_soup3_perturb0.9 runs deleted. findings.md updated.

## 2026-10-16 12:20 — Release: InternVL average rebuilt without hadikp members
- Found before publishing: S_soup10_vision includes ZH_hk241, ZH1_hk241_seed1, ZH2_hk241_seed2 (trained with 241 hadikp
  pages, licence unknown); the model card promises released weights without them. SmolVLM and granite averages use only
  labels_train_v6.
- S_soup4_clean (1608049) = Z_aug_vision ckpt-528, Z1_aug_vision_seed1, Z2_aug_vision_seed2, Z3_aug_vision_seed3 ckpt-440:
  0.734 / 0.712 vs S_soup10_vision 0.728 / 0.732. Test scoring 1608124 (test.slurm fixed: it wrote its log before creating
  the run folder).
- User chose to retrain 6 clean seeds: ZR_seed4–6 (1608125–27), ZR_seed7–9 (1608128–30, after the first three), default
  recipe configs/internvl3-1b.toml. Then a 10-member clean average with a fresh test score; released weights in bf16
  (all reported GPU scores were computed with bf16 weights).
- 10:57 S_soup4_clean test (1608124): strict 0.726, lenient 0.739, content 0.841; repaired strict 0.732, lenient 0.746,
  content 0.849. S_soup10_vision test was lenient 0.745 (0.746 repaired): the 4 clean members alone lose 0.006 raw,
  nothing repaired, within the seed spread. The 10-member clean average is still the planned release.
- 11:50 ZR_seed5 / ZR_seed6 done (1 h each), val chose epoch 5 for both: gold 0.698 / 0.701, val 0.700 / 0.715 lenient
  (single-model range). ZR_seed4 still running; ZR_seed7–9 wait on it (afterany:1608125).
- 11:55 ZR_seed4 done: val chose epoch 5, gold 0.724 / val 0.713. ZR_seed7 running, ZR_seed8–9 pending.
- 12:05 Released ResuVL-S (granite-docling average of 4) and ResuVL-M (SmolVLM average of 3) to huggingface.co/furkankucuk, bf16, CC BY-NC 4.0 + LICENSE-base-model.txt (Apache 2.0). Name chosen by the user from 10 options checked on HF / GitHub / PyPI. parse --model furkankucuk/ResuVL-{S,M} tested from an empty cache.
- 13:00 ZR_seed7 done: val chose epoch 6, gold 0.721 / val 0.711. ZR_seed9 done: val chose epoch 5, gold 0.726 / val 0.709. ZR_seed8 still running.
- 13:05 ZR_seed8 done: val chose epoch 5, gold 0.695 / val 0.696. Submitted S_soup10_clean (1608784) = the 4 S_soup4_clean members + ZR_seed4–9 at val-chosen epochs (seed7 ep6 = model, others checkpoint-440 = ep5), then its test scoring S_soup10_clean_test (1608785, afterok).
- 13:30 S_soup10_clean (1608784) dev 0.732 / 0.724 lenient, content 0.847 / 0.855. Test (1608785): strict 0.722, lenient 0.735, content 0.840; repaired 0.725 / 0.739 / 0.845; unparsed 4 of 204 (2 token limit). vs S_soup10_vision test 0.745 (0.746), S_soup4_clean 0.739 (0.746). Release = S_soup10_clean as planned before the test score (dev ties S_soup4_clean: 0.732 / 0.724 vs 0.734 / 0.712); choosing between them on test would select on the test pages.
- 13:40 Release check: the bf16 release folders, scored again on the 204 test pages (REL_L/M/S_test, 1608852–54), reproduce the published scores exactly: L 0.735 (0.739), M 0.677 (0.683), S 0.633 (0.647). Answers identical on 204 / 204 (L, M); S 192 / 204, the other 12 are token-limit runaways cut earlier (lower --max-new-tokens in the check), scored 0 in both. Uploaded S/M weights have the same sha256 as the scored folders.
- 14:40 Uploaded ResuVL-L (S_soup10_clean, bf16, sha256 4c60c2cd…) to huggingface.co/furkankucuk/ResuVL-L; parse --model furkankucuk/ResuVL-L tested from an empty cache. Docs updated to its scores.
- 14:50 Published GitHub release v0.1.0 (tag on f230c11). Obsolete weights on TRUBA listed for the user to delete: S_soup10_vision/model, S_soup4_clean/model, SV_soup3_tiles20/model, SV_tiles20_seed42/1/2 checkpoints.

## 2026-10-16 16:30 — Literature search: paths to F1 > 0.9 under 1B
- Four research agents (base models, larger-model transfer, data/labels, objectives/architecture) worked from one
  context file; reports in docs/research/literature-2026-10-16/ (unedited, sources inside). Claude verified on the HF
  API: Qwen3.5-0.8B 873M, LightOnOCR-3-0.8B 853M, Xiaomi-OCR-0 873M, OvisOCR2 853M (all Qwen3_5ForConditionalGeneration,
  Apache-2.0); PaddleOCR-VL-1.6 959M; LightOnOCR-2-1B 1,006M (over the cap); InternVL3.5-1B 1,061M (over); GLM-OCR
  1,325M (over, despite "0.9B"). InternVL3-1B/2B/8B share vocab 151,674 and the same vision config (logit KD possible).
- Common conclusion: F1 tracks language-model size (165M 0.633 < 360M 0.677 < 0.5B 0.735 < 1.5B 0.754 < 7B 0.824);
  omission is a decision error (56 % of missed values appear in one of 8 samples), not a reading error.
- Ranked hypotheses: (1) Qwen3.5-0.8B-family base (larger LM inside 1B; OCR-trained variants), (2) logit distillation
  from a fine-tuned InternVL3-8B, (3) synthetic French CVs with exact labels, (4) measure the label-noise floor (5
  teacher runs + blind adjudication). Not pursued: pruning 2B->1B (61 % LM cut), RL-only (no sub-1B report), encoder
  swap (needs ~29M alignment pairs), transcribe-then-structure, extra decoder layers, multi-token prediction.

## 2026-10-16 17:50 — Three new hypotheses running (from the literature search)
- H-base (larger LM inside ~1B): N35_seed42/1/2 (1608960–62) = InternVL3.5-1B-HF (Qwen3-0.6B LM, 1.06B total) with the
  unchanged configs/internvl3-1b.toml recipe; compare with the 1B single-seed runs (mean 0.711 / 0.707). Qwen3.5-0.8B
  family (LightOnOCR-3-0.8B, Xiaomi-OCR-0): code path works under transformers 5.5.4 (venv/tf5_pkgs, smoke test ok:
  mask check, train, save, load, generate), but without flash-linear-attention / causal-conv1d kernels it trains at
  36 s per 16 pages vs 3.9 s for InternVL3.5 (6 epochs = 5.3 h); Q35_seed42 cancelled; kernels need approval.
- H-distil: T8B_lora (1608972, 20:00) = InternVL3-8B teacher, LoRA r32 + projector, lr 1e-4, 3 epochs, augment 0.5,
  merged to bf16, scored on gold-50 + val. train.py now has --lora (teachers) and --teacher/--kd-weight/--kd-temp
  ((1-w) CE + w T^2 KL(teacher||student) on answer tokens; checked: 0 for equal logits, equals manual KL).
- H-synth: experiments/synth_cvs.py renders training records (not val/gold) with new name/contacts, sometimes swapped
  sections, 4 layouts x 9 fonts x 9 colours x 4 date styles, every label value printed, 120 dpi; 3,000 pages
  (68 skipped: over one page). SY1k_seed42/1/2 (1608974–76, 20:00) = real + 1,000 synthetic; SY3k_seed42/1/2
  (1608977–79, after them) = real + 3,000. Same recipe, 6 epochs. Disk 1,958 GiB of 2,000 at submit.
- 18:40 H-base result, InternVL3.5-1B with the InternVL3 recipe: val-chosen seed42 ep6 0.721 / 0.703, seed1 ep4
  0.723 / 0.707, seed2 ep5 0.691 / 0.715; mean 0.712 / 0.708 vs InternVL3-1B single runs 0.711 / 0.707 (ZR seeds 4-9).
  No gain: the larger Qwen3-0.6B LM (vs Qwen2.5-0.5B) does not help at this recipe, and the model is 1.06B. Not adopted
  as is; checkpoints kept for a possible lr check or soup.
- 18:50 T8B_lora (1608980, 54 min): InternVL3-8B LoRA r32 3 epochs, gold 0.780 / val 0.786 lenient (vs 1B singles 0.711 / 0.707). Submitted KD8_seed42/1/2: InternVL3-1B recipe + --teacher runs/T8B_lora/model, kd-weight 0.5, T 2.
- 19:10 H-distil result (not adopted): KD8_seed42/1/2 (1609007–09, ~1 h 20 min each; 7.3 s/step with the 8B teacher
  in the loop). Val-chosen: seed42 ep6 0.704 / 0.717, seed1 ep6 0.697 / 0.715, seed2 ep4 0.718 / 0.705; mean 0.706 /
  0.712 vs InternVL3-1B singles 0.711 / 0.707 (-0.005 / +0.005, inside the seed spread). Off-policy logit KD from a
  teacher that scores 0.780 / 0.786 does not move the student at weight 0.5, T 2. Possible reasons: the student
  already fits its training pages (train 0.91 vs dev 0.70), so a softer target on the same pages adds no new
  information; the gap is generalisation, which KD on the same 1,472 pages cannot close. Next for this line: KD on
  pages the student has not seen (synthetic or unlabelled pages, teacher answers as targets), not more weight sweeps.
- 18:55 Kernels for Qwen3.5: flash-linear-attention / fla-core 0.4.2 (0.5.x needs torch >= 2.7; shared torch is 2.6)
  and causal-conv1d 1.7.0 built from source (CUDA 12.4, old C++ ABI) in venv/tf5_pkgs. Speed 4.8 s/step after kernel
  compilation vs 36 s without. Q35_seed42/1/2 (1609048–50, 20:00). LightOnOCR-3 and Xiaomi-OCR-0 share the Qwen3.5
  architecture, chat template and vocabulary but not the tokenizer (different merges and pre-tokenizer), so train.py
  now saves the processor in every checkpoint and soup.py copies it (commit b05e0e8; checked: average tokenizes like
  LightOnOCR-3). LO3_seed42/1/2 (1609057/59/61) after the Q35 runs; XO0_seed42/1/2 (1609058/60/62) after LO3.
- 20:35 H-base result, Qwen3.5-0.8B (873M; Gated DeltaNet + attention LM, 248k vocab) with the unchanged
  configs/internvl3-1b.toml recipe (lr, epochs, augment), max-patches 12, transformers 5.5.4 + fla kernels: ADOPTED as
  the new InternVL-track candidate. Q35_seed42/1/2 (1609048–50, ~1 h 17 min each). Val-chosen: seed42 ep6 0.849 /
  0.800, seed1 ep5 0.829 / 0.821, seed2 ep6 0.826 / 0.797; mean 0.835 / 0.806 vs InternVL3-1B singles 0.711 / 0.707
  (+0.12 / +0.10, ~6x the seed spread, on both dev sets). Every epoch >= 4 is >= 0.826 on gold. Above the 8B LoRA
  teacher (0.780 / 0.786) and above the released 10-run InternVL average (0.732 / 0.724). Checked: rescored locally
  (same numbers; strict 0.816 / 0.811, content 0.929 / 0.905 for seed1 ep5); pages and order match the gold file;
  answers are model output (seed1 page 1 reads the name and title, invents a phone number). Gold-50 seed1 ep5: 43
  valid, 6 schema errors, 1 unparsed. Likely mechanism (not tested): a newer and larger LM (Qwen3.5 0.8B, trained on
  far more multilingual and document data than Qwen2.5-0.5B) inside the same size budget; the research ranking put
  this first. Next: Q35_soup3 (1609075) = average of the 3 val-chosen checkpoints, scored on dev; Q35_soup3_test
  (1609076) on the test set. LightOnOCR-3 and Xiaomi-OCR-0 (same architecture, OCR-trained) running/queued.
- 20:55 H-synth result, 1,000 synthetic pages (InternVL3-1B): SY1k_seed42/1/2 (1608974–76, ~1 h 22 min each). Val-chosen:
  seed42 ep6 0.740 / 0.715, seed1 ep6 0.747 / 0.716, seed2 ep6 0.736 / 0.742; mean 0.741 / 0.724 vs singles 0.711 /
  0.707 (+0.030 / +0.017). Gold beats the seed spread; validation is at its upper edge. All three chose epoch 6,
  and the runs take 882 steps in total instead of 528 (2,472 vs 1,472 pages), so more optimisation steps are a confound:
  likely gain, not yet separated from training length. SY3k (3,000 synthetic, 1608977–79) running. If it is
  adopted, the next run is Qwen3.5-0.8B + synthetic pages.
- 21:10 Q35_soup3 (1609075): uniform average of Q35_seed42/model, Q35_seed1/checkpoint-440, Q35_seed2/model. Dev 0.858 /
  0.822 lenient (strict gold 0.844, content gold 0.952) vs member mean 0.835 / 0.806 (+0.023 / +0.016) and vs the
  released InternVL average 0.732 / 0.724. Test scoring Q35_soup3_test (1609076) running.
- 21:35 MILESTONE test, Q35_soup3_test (1609076): 204 test pages, lenient 0.846 (repaired 0.846), strict 0.836,
  content 0.916; 185 valid, 17 schema errors, 2 unparsed, 0 at the token limit. Released ResuVL-L (InternVL3-1B
  10-run average): 0.735 (0.739), strict 0.722, content 0.840, 4 unparsed. +0.111 lenient, +0.076 content, at 0.87B.
  GPU generation (TRUBA, batch 8, dev pages): ~25 % fewer tokens/s than InternVL3-1B, ~10 % shorter answers in tokens;
  CPU speed not measured (needs transformers >= 5.2 on the Mac; no fla kernels on CPU).

## 2026-10-16 21:40 — Release 0.2.0: Qwen3.5-0.8B average becomes ResuVL-L (user: "our new spotlight model")
- User chose: replace furkankucuk/ResuVL-L (old InternVL3-1B weights kept as HF revision v0.1.0); docs first, then
  publish. Checked before release: Qwen3.5-0.8B is Apache 2.0; Q35 runs trained on labels_train_v6.jsonl only (0
  hadikp pages).
- bf16 release file runs/release_bf16/ResuVL-L-v2 (from Q35_soup3/model; ../make_bf16.py in a job, the login node hit
  its CPU limit). REL_L2_test (1609092): lenient 0.845 (repaired 0.845), strict 0.835, content 0.916, 2 unparsed;
  172 of 204 answers identical to the unrounded average (0.846). Docs report the release file.
- Package moved to transformers 5.5.4 (pyproject pin, version 0.2.0). Same released files under 5.5.4 on the test
  pages: TF5_L_test (InternVL3-1B average) 204/204 identical answers, 0.735; TF5_M_test (SmolVLM) 62/204 identical,
  0.677 -> 0.681 raw, 0.683 -> 0.682 repaired; TF5_S_test (granite-docling) 50/204 identical, 0.633 -> 0.652 raw,
  0.647 -> 0.653 repaired, unparsed 16 -> 13, token limit 12 -> 7. Idefics3ImageProcessor became
  Idefics3ImageProcessorPil in 5.x (same size settings printed); cause of the differences not isolated. Docs give the
  5.5.4 test scores for S and M (what package users get) with the 4.57.6 scores next to them; dev scores stay 4.57.6.
- 21:45 H-base, LightOnOCR-3-0.8B (Qwen3.5 architecture + OCR training, own tokenizer) with the same recipe: LO3_seed42/1/2
  (1609057/59/61, ~1 h 15 min each). Val-chosen: seed42 ep6 0.844 / 0.812, seed1 ep4 0.807 / 0.813, seed2 ep4 0.838 /
  0.827; mean 0.830 / 0.817 vs Qwen3.5-0.8B 0.835 / 0.806 (-0.005 / +0.011, inside the seed spread). OCR training adds
  nothing measurable at this recipe. LO3_soup3 (average of the three) scored on dev for comparison with Q35_soup3.
- CPU speed of ResuVL-L v2 on the M2 Max (transformers 5.5.4, float32, batch 1, no fla kernels on CPU: torch fallback
  uses ~1.3 cores): first validation page not finished after 24 min, vs 35–80 s per page for the InternVL3-1B average.
  Release choice asked of the user.
- 21:55 LO3_soup3 (1609120): average of the three LightOnOCR-3 runs, dev 0.847 / 0.829 vs Q35_soup3 0.858 / 0.822
  (-0.011 / +0.007, a tie within the spread). Qwen3.5-0.8B stays the release (chosen first, Apache 2.0, no tie-break
  on test pages). LO3 weights kept for a possible cross-base comparison; not tested on the test pages.
- 22:25 H-synth, 3,000 synthetic pages (InternVL3-1B): SY3k_seed42/1/2 (1608977–79, ~2 h each). Val-chosen: seed42 ep5
  0.764 / 0.748, seed1 ep6 0.745 / 0.736, seed2 ep6 0.741 / 0.743; mean 0.750 / 0.742 vs singles 0.711 / 0.707
  (+0.039 / +0.035, above the spread on both sets) and vs SY1k 0.741 / 0.724 (+0.009 / +0.018). More synthetic pages,
  more gain, but these runs take ~1,650 steps instead of 528, so training length is still confounded. Control queued:
  ZL10_seed42/1/2 = real pages only, 10 epochs (880 steps, matches SY1k), after the Q35SY1k runs.
- 22:50 H-base, Xiaomi-OCR-0 (Qwen3.5 architecture, OCR model writing structured JSON): XO0_seed1 (1609060) ep6 0.539 /
  0.478; 44 of 79 val answers unparsed, every one at a spot where the model wrote `":[{"` in place of a word inside
  copied text (`des ventes par` -> `des":[{" par`, `diplômée` -> `":[{"ée`). Likely its pre-training output format;
  not adopted with this recipe. Scoring of ep3 had crashed: json.loads raised RecursionError on Python 3.11 for an
  answer with thousands of "[" (fixed in metrics.py, commit baa7dfd).
- 22:55 CPU speed of ResuVL-L v2 explained: torch.profiler on 10 generated tokens: 86 % of CPU time in
  aten::_slow_conv2d_forward, 1.1 M calls. The depthwise conv1d (kernel 4, ~6,100 channels) of the 18 linear-attention
  layers runs one channel at a time on PyTorch's macOS CPU build (mkldnn not available): 224 ms per call. generate.py
  now replaces it, when the causal-conv1d kernel is missing, with 4 shifted multiply-adds (0.05 ms; same argmax, max
  logit diff 1.5e-4 on one page; tests/test_generate_conv.py). Small page: prefill 13.5 s -> 1.9 s, ~0.12 s per token.
  XO0_seed42 and XO0_seed2 cancelled (22:50): the failure in seed1 is systematic; their GPUs go to Q35SY1k.
- 23:55 CPU speed test of ResuVL-L v2 with the depthwise fix (same 10 val pages, M2 Max, float32, batch 1, transformers
  5.5.4): median 127.5 s/page (67–205), 7.1 tokens/s, peak RAM 5.8 GB, 10/10 valid, F1 0.816 on these pages. InternVL3-1B
  average on 2026-10-14: 62 s, 15.6 tokens/s, 4.9 GB. About 2x slower on CPU; docs say 128 s.
- 00:20 Published: furkankucuk/ResuVL-L tagged v0.1.0 at the InternVL3-1B commit (70339aa684), then replaced by the
  Qwen3.5-0.8B bf16 average (commit 2d43574cc4, sha256 d0fb3069…; old InternVL processor files removed). ResuVL-S and
  ResuVL-M cards updated. parse --model furkankucuk/ResuVL-L tested from an empty cache on CPU: valid record.

## 2026-10-17 00:15 — Qwen3.5-0.8B + 1,000 synthetic pages: no measurable gain
- Q35SY1k_seed42/1/2 (1609084–86, ~1 h 40 min each), configs/internvl3-1b.toml recipe on train_v6_synth1k.jsonl.
  Val-chosen: seed42 ep6 0.833 / 0.804, seed1 ep5 0.855 / 0.810, seed2 ep6 0.840 / 0.823; mean 0.843 / 0.812 vs Q35
  0.835 / 0.806 (+0.008 / +0.006, inside the spread). The synthetic pages helped InternVL3-1B (+0.030 / +0.017) but not
  Qwen3.5; one reading: the larger base already covers what the rendered pages teach.
- Queued: Q35SY_soup3 (1609734) and Q35_soup6 (1609735, all six Qwen3.5 runs; same base, so averaging applies) on
  dev; Q35lr2_seed42/1/2 (1609736/38/40) = lr 2e-5 (the recipe was never tuned for Qwen3.5); Q35p20_seed42/1/2
  (1609737/39/41, after them) = max-patches 20 (4.0 Mpx cap instead of 2.4; tests whether reading errors fall).
- 00:35 Q35SY_soup3 (1609734) dev 0.875 / 0.820; Q35_soup6 (1609735, all six Qwen3.5 runs) 0.873 / 0.820; released
  Q35_soup3 0.858 / 0.822. Gold +0.017 / +0.015, validation -0.002: gains on one set only, so neither replaces the
  release (rule: beat the spread on both sets). Learning-rate runs Q35lr2 started 00:30.
- 01:25 CPU speed, ResuVL-L v2 (user asked for 8-bit quantisation): torch dynamic int8 (qnnpack, the only engine on
  macOS) on the whole model: 0.082 vs 0.100 s per token on a small page, prompt 2x slower, answers depart from float32
  after 10–17 tokens; not adopted. Decode profile: 77 % of token time in aten::mm (memory-bound: ~3.5 GB fp32 weights
  read per token, ~1 GB of it the 248k-vocab output layer). bf16 (M2 has FEAT_BF16): 37 vs 102 ms per token, prompt
  6x slower when the whole model is bf16. Language model + lm_head in bf16, vision encoder fp32: 10 val pages median
  58.5 s (wrapper) and 54.5 s (default path) vs 127.5 s, F1 0.811 vs 0.816 (827 vs 826 matched values), 10/10 valid.
  generate.py now does this on CPUs with bf16 instructions (cpu_has_bf16). Released as 0.2.1.
- 02:20 Training-length control ZL10_seed42/1/2 (1609192–94; InternVL3-1B, real pages only, 10 epochs = 880 steps,
  ~1 h 35 min each). Val-chosen: seed42 ep10 0.724 / 0.732, seed1 ep8 0.725 / 0.721, seed2 ep9 0.700 / 0.717; mean
  0.716 / 0.723. vs SY1k (882 steps) 0.741 / 0.724: synthetic pages add +0.025 gold, +0.001 val at matched length.
  The SY1k gain was mostly training length; not adopted. SY3k (0.750 / 0.742, ~1,650 steps) has no matched control.
  docs/findings.md corrected. Q35lr2 seed1 ep4 0.835 / 0.798, seed2 ep6 0.843 / 0.798 so far (seed42 pending).
- 02:35 Q35lr2_seed42/1/2 (lr 2e-5, 1609736/38/40) done. Val-chosen: seed42 ep6 0.841 / 0.802, seed1 ep4 0.835 / 0.798,
  seed2 ep6 0.843 / 0.798; mean 0.840 / 0.799 vs lr 1e-5 0.835 / 0.806 (+0.005 / -0.007): no gain, lr 1e-5 stays.
  Queued Q35e10_seed42/1/2 = Qwen3.5-0.8B, 10 epochs (ZL10 showed InternVL3-1B gains +0.005 / +0.016 from 6 -> 10
  epochs on real pages), min scored epoch 5.
