# D. Objectives, RL, vision and architecture inside one <1B model (2026-10-16)

## Bottom line
Sub-1B document RL papers show +1.5-2 points on top of huge SFT; that is inside our seed spread, which matches our null GRPO. Vision/resolution changes target misreads (2.9 %), not omissions (10.6 %). The best-evidenced levers left are capacity transfer (distillation from the 8B we already trained) and a stronger pretrained base. Details below. Inferences are marked "(inference)".

Facts from the repo (checked): the journal's GRPO probe found best-of-8 0.768 vs greedy 0.703 on gold, and 56 % of greedy-missed values appear in at least one of 8 samples. So omission is a decision problem, not a "cannot read it" problem. Unseen-page GRPO (lenient and content reward, group 4, lr 3e-6) gave 0.708/0.714 and 0.718/0.712 vs start 0.705/0.719 (docs/research/journal.md, 2026-10-15 22:10). On test, S_soup10_clean leaves only 4/204 pages unparsed and repair adds ~0.003 (journal 2026-10-16 13:30), so the dev "6.7 % not parsed" is 3 pages.

Verified this session: InternVL3-1B, 2B and 8B share vocab_size 151,674 and an identical vision config (1024 wide, 24 layers, 448 px, downsample 0.5) (HF configs: https://huggingface.co/OpenGVLab/InternVL3-1B-hf/raw/main/config.json, .../InternVL3-8B-hf/raw/main/config.json).

## 1. RL for OCR / parsing / extraction
| Work | Model | Recipe | Gain | URL |
|---|---|---|---|---|
| olmOCR 2 | Qwen2.5-VL-7B | 1 epoch SFT, then 1 epoch GRPO on 2,186 synthetic pages (30,381 binary unit tests), 28 completions, beta 0.01, reward = fraction of tests passed + EOS + metadata rewards, 6 RL seeds averaged | olmOCR-Bench 78.5 to 82.4 (tables +12.0) | https://arxiv.org/html/2510.19817v1 |
| Infinity-Parser | Qwen2.5-VL-7B | GRPO, group 8, beta 0.01, lr 1e-6, 43K docs, reward = edit distance + paragraph count + reading order | RL-only (no SFT) EN edit 0.141 vs SFT 0.198; SFT+RL 0.163 | https://arxiv.org/html/2510.15349v1 |
| HunyuanOCR | ~1B (ViT 0.4B + LM 0.5B; HF safetensors 1.12B) | GRPO group 8, lr 8e-7, KL 0, pass-rate and zero-variance filtering, schema-breaking output = reward 0 | OmniDocBench 92.5 to 94.1, IE about +2 | https://arxiv.org/html/2511.19575v1 |
| LightOnOCR | 1B | GRPO 1 epoch, lr 4e-5, beta 0.01, 28 rollouts, olmOCR unit tests + compression-based loop penalty + EOS check | v1 76.1 to 77.7 olmOCR-Bench; loopy outputs 1.14 % to 0.50 % | https://arxiv.org/html/2601.14251 |
| GLM-OCR | 0.9B claimed (see below) | GRPO after SFT; KIE reward = field-level F1 + JSON parse check + missing/duplicate-field penalty; rollouts from the SFT model, stratified by difficulty | KIE 93.7 Nanonets-KIE; no RL ablation | https://arxiv.org/html/2603.10910v1 |
| ReceiptBench | InternVL3-2B, Qwen3-VL-4B/8B | SFT then GRPO, 16 samples, beta 0.01, per-field reward, FP penalty -0.5 | InternVL3-2B 0.650 to 0.447 (worse); 4B +0.006; 8B +0.016 | https://arxiv.org/html/2605.22413 |
| Medical VIE | Qwen2.5-VL-7B | 100 images, reward = alpha-weighted precision/recall of key-value pairs, group 8, beta 0.04 | F1 77.81 vs JSON-SFT on the same 100 images 66.69 | https://arxiv.org/html/2506.13363 |

What successful works do differently from us:
- Fresh prompts with exact or presence-style rewards (olmOCR 2 renders HTML and writes tests; ours rewarded teacher splits, which flipped sample order in 24 % of pairs).
- Larger groups (8-28 vs our 4), often KL 0 to 0.01.
- Difficulty filtering: drop groups with zero reward variance or trivial/unsolvable pass rate (HunyuanOCR; DAPO dynamic sampling, https://arxiv.org/pdf/2503.14476). Our journal lists "keep only groups where a sample beats greedy" as untested; TRL's docs say dynamic sampling is not supported (https://github.com/huggingface/trl/blob/main/docs/source/paper_index.md), so filter offline.
- Loop and validity terms inside the reward (LightOnOCR, GLM-OCR).
- Multi-seed averaging of RL runs (olmOCR 2).

RL-only for sub-1B: I found no report. RL-only works at 7B when the base already emits near-target output (Infinity-Parser). Our base cannot emit the schema, so groups would be all-zero (inference). Small-VLM RL papers all SFT first (CLGRPO 1B, https://arxiv.org/abs/2506.18048; SmolVLM2-500M GRPO paper https://arxiv.org/html/2609.28757, which gives no isolated GRPO ablation on OCR).
Verdict: gains at about 1B are +1.4 to +2 points on top of huge SFT, i.e. our noise level. ReceiptBench shows the same family (InternVL3-2B) can collapse. Not a first bet.

## 2. Recall / omission objectives
- Omission-contrastive training: Yang et al. 2019 train the model to give the reference higher probability than a copy with words deleted; reported fewer omissions on three language pairs, converges in hundreds of steps, only objective and data change (https://aclanthology.org/P19-1623/). Transfer to JSON is untested. Our DPO on the model's own wrong answers got worse; minimal-pair negatives (gold with one value or list item deleted) differ from that. Caveat (inference): a shorter sequence has higher likelihood, so use per-token normalisation and a margin. Effort: about 60 lines in train.py (second forward), 1.6x step time.
- Bag-of-words auxiliary target (https://arxiv.org/pdf/1805.04871): its stated motivation is multiple valid outputs, not omission; weak fit.
- Transcribe-then-structure in one generation: I found no paper isolating its effect on F1 for VLM extraction. Our own transcript pre-stage and key-by-key runs gave nothing; doubling output length doubles CPU time. Skip.
- Loss weighting on value tokens: no direct evidence found. JSON syntax tokens reach near-zero loss after epoch 1, so they already carry little weight (inference). A 20-minute diagnostic (loss share by token type on dev) would settle it.
- Auxiliary MTP loss: Gloeckle et al. report MTP is worse than next-token at small sizes (300M+ range) (https://arxiv.org/pdf/2404.19737). Not for accuracy.

## 3. Vision side
- Evidence exists for native resolution and OCR pretraining: PaddleOCR-VL (NaViT-style encoder from Keye-VL, 2-layer MLP, ERNIE-0.3B, stage 2 max 2048x28x28 px, all weights trainable, olmOCR-Bench 80.0) https://arxiv.org/html/2510.14528v3; HunyuanOCR trains the ViT from stage 1; LightOnOCR raised max edge 1024 to 1540 px.
- None isolates resolution with an ablation. Our pages are 120 dpi, about 992x1403 px; 12 tiles already upsample them, and misreads are 2.9 % of values. Expect little.
- Encoder/LM balance: SmolVLM finds that with a 360M LM, a 428M encoder adds 11.6 % but costs 66 % more parameters; compact encoders preferred for small LMs (https://arxiv.org/html/2504.05299). Pixel-shuffle r=4 helps small models.
- Our own table (inference): LM size tracks F1 (Granite-165M 0.633, SmolLM2-360M 0.677, Qwen2.5-0.5B 0.735, 1.5B 0.754 LoRA, 7B 0.824 LoRA) even though granite has a doc-trained encoder. So spend budget on the LM, not the encoder.
- Vocabulary pruning frees budget: LightOnOCR Table 4: 151k vocab 1005.6M params, 32k 883.6M (96 % of base score on a Chinese-inclusive benchmark), 16k 866.8M. For Latin-script CVs the loss should be smaller (inference).
- Candidate bases (safetensors headers read this session):
  - LightOnOCR-2-1B: 1,005.6M (ViT 403M, Qwen3-0.6B LM layers 440M, embeddings 156M tied), Apache-2.0, French in training. Over 1B by 5.6M; must prune vocab. Requires its own venv. https://huggingface.co/lightonai/LightOnOCR-2-1B
  - PaddleOCR-VL: 958.6M, Apache-2.0, LM layers only 255M. https://huggingface.co/PaddlePaddle/PaddleOCR-VL
  - GLM-OCR: marketed 0.9B but safetensors sum is 1,325M (ViT 434M, LM layers 709M incl. 1 MTP layer, embedding 91M + untied head 91M). Violates the cap even without MTP (about 1.28B). Excluded. https://huggingface.co/zai-org/GLM-OCR
  - HunyuanOCR: 1.12B and "other" licence. Excluded.
- Swapping an OCR encoder into InternVL3 with projector re-alignment needs large alignment data (PaddleOCR-VL stage 1 used 29M pairs); not feasible at 1.4k pages.

## 4. Output side
- Constrained decoding speed: XGrammar reports under 40 us per token mask for JSON schema, and TPOT 6.2 to 6.3 ms with vs without it on Llama-3.1-8B (https://arxiv.org/html/2411.15100). It has a Hugging Face logits-processor integration (I could not open the docs page; see Could not verify). It would remove the lm-format-enforcer slowdown.
- Upside for F1 is small: lm-format-enforcer gave the same F1, and on test only 4/204 pages are unparsed. Grammars cannot stop semantic loops. Loops are better handled by training: DITTO pseudo-repetition training (https://arxiv.org/abs/2206.02369) or a repetition term in an RL reward (LightOnOCR).
- MTP / parallel decoding is a speed tool: GLM-OCR 5.2 tokens/step, about 50 % throughput (https://arxiv.org/html/2603.10910v1); PTP on Qwen2.5-VL-3B 1.6x-2.2x with one text edit distance 0.0585 to 0.0431 (PTP-1) but PTP-2 equals NTP (https://arxiv.org/html/2603.15206). No sub-1B evidence.
- Extra decoder layers: 0.5B to 1.5B LM gave only +0.02 here, so +8 layers from freed vocab budget would likely be inside noise (inference).

## 5. Longer / better supervised training
- PaddleOCR-VL: 2 epochs on 2.7M samples, low LR 5e-6, augmentation on. LightOnOCR: loss masked to assistant tokens, 5-checkpoint averaging, empty-page examples, augmentation p 0.22. LightOnOCR merging gave +0.6 (soup α 0.4); we already soup (+0.02).
- Capacity transfer: Qwen3 strong-to-weak distillation (0.6B to 14B students; off-policy then on-policy KL). On Qwen3-8B: RL 17,920 GPU h reached AIME'24 67.6; on-policy distillation 1,800 GPU h reached 74.4; RL gave no pass@64 gain (https://arxiv.org/html/2505.09388). GKD trains on the student's own samples scored by the teacher (https://arxiv.org/abs/2306.13649). TRL has `DistillationTrainer` documented for VLMs (reverse-KL at beta=1) (https://github.com/huggingface/trl/blob/main/docs/source/distillation_trainer.md).
- Label smoothing, curriculum, hard-page weighting: no small-VLM document evidence found; with a 0.91 train vs 0.70 dev gap they address fit, not generalisation (inference).

## Ranked ideas (effort, compute, constraint)
| # | Idea | Targets | Effort | Compute | One-model rule |
|---|---|---|---|---|---|
| 1 | Logit distillation from InternVL3-8B into the 1B | capacity gap (8B +0.09), loops | 40 lines for teacher-forced KL; on-policy via TRL | teacher forward 8B per step, about 3-4 h/run | OK; teacher only at training, but CLAUDE.md needs owner approval for a teacher role |
| 2 | LightOnOCR-2-1B base, vocab pruned under 1B | OCR/French pretraining, native-res encoder, loop rate | about 1 day (vocab remap, new venv) | 1-2 h/run | OK after pruning |
| 3 | Omission-contrastive auxiliary loss | omissions 10.6 % | about 60 lines | 1.6 h/run | OK |

RL redo is not ranked: reopen only with fresh exact-label pages (for example rendered synthetic CVs), group 16-28, offline difficulty filtering, KL 0-0.01, repetition/EOS terms, multi-seed average.

## First experiment (idea 1) vs baseline 0.732 / 0.724
1. Diagnostic, 30 min: mean probability the 8B teacher gives the gold token on training pages vs dev pages. If it is at least 0.99 on training pages, teacher-forced KD equals CE; use an early-epoch teacher checkpoint or an out-of-fold teacher instead.
2. Arms (configs/internvl3-1b.toml recipe unchanged): CE only (existing single-seed runs, gold 0.695-0.726, val 0.696-0.715, as baseline distribution) vs CE + 0.5 x T²·KL(teacher||student), T=2, teacher logits computed online on the same augmented batch. 3 seeds, val-chosen epoch, report gold-50 and val.
3. Success: 3-seed mean at least +0.02 over the baseline mean on both dev sets, with unparsed pages not up. If it passes, run on-policy (lmbda 1, beta 1) on pages the teacher has not memorised.
Cost: about 3 seeds x 3.5 h = 11 GPU-h.

## Could not verify
- GKD and DeepSeek-R1 appendix numbers (page text truncated); only the abstract-level claim for GKD is verified.
- XGrammar HF-integration docs page (404) and llguidance speed; JSON-schema maxItems/maxLength support unconfirmed.
- Exact TRL version that introduced `DistillationTrainer` (context7 lists v1.0.0; local venv has no trl).
- Yang et al. 2019 omission-rate numbers; Gloeckle et al. per-size numbers.
- GLM-OCR RL hyperparameters and ablations; HunyuanOCR RL sample count; summaries came from a fetch tool and were not cross-read in the PDFs.
- Whether LightOnOCR-2 pruned to 32k works for French CVs, and its transformers version needs.
- SmolDocling / granite-docling training recipe details (not fetched); SigLIP2 NaFlex OCR evidence (not fetched).
