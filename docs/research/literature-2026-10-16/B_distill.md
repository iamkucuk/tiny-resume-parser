# B. Using a larger model at training time (research report, 2026-10-16)

References in [n] map to the Sources list at the end. "J" = docs/research/journal.md, "F" = docs/findings.md, "C" = CLAUDE.md of this project. GPU-hour figures marked (est.) are my arithmetic, not measurements.

## Ranked recommendation
1. **Token-level logit distillation from an InternVL3-8B teacher fine-tuned on our labels, into the existing 1B full-fine-tune recipe.** Stage a: cross-entropy on gold plus forward KL to the teacher on the same augmented page. Stage b: one on-policy stage (student samples, teacher scores every token) on pages without labels. Expected gain +0.01 to +0.03 F1 (est., from [5][6][15]); the ceiling is the teacher's 0.82, so 0.9 is not reachable this way.
2. **More in-domain pages with teacher-written or synthetic labels (noisy student).** Limited by data supply, and by our own tests (below).
3. **Hidden-state / relation distillation (LLaVA-KD, Align-KD)** only as an add-on if 1 shows a gain.

Not recommended: pruning a 2B-4B VLM to under 1B; merging or weight-selection tricks.

## 0. Facts that decide feasibility
- All InternVL3 sizes (1B, 2B, 8B, 14B, 38B, HF format) have text `vocab_size` 151,674 and a Qwen2 text model; 1B/2B/8B/14B use a 24-layer, 1,024-wide InternViT with 448 px tiles [19]. One processor output (input_ids, pixel_values) can feed teacher and student, and logits line up position by position. Hidden size is 3,584 (8B) vs 896 (1B) [19].
- Parameter budget (my arithmetic from the configs, reproduces the 0.94B figure): ViT about 0.30B, LLM body 358M, embedding 136M plus untied head 136M [19]. Embedding plus head is 29% of the shipped model; the head is 28% of the LLM's per-token multiply-adds.
- TRL `GKDTrainer` is text-only: `compute_loss` calls `model(input_ids, attention_mask)` and checks equal vocab size [17]. The newer stable `DistillationTrainer` supports VLMs (tested on Qwen2-VL, Qwen2.5-VL, Gemma 3, LLaVA-NeXT; vLLM generation; prompt-only data, student samples) but InternVL is not on its tested list [18].
- Project gaps: 1B average 0.732/0.724; 2B LoRA 0.754/0.716; 8B LoRA 0.824/0.812 (F L122). Labelling model run twice: 0.912 against reviewed labels. Student on training pages 0.910, unseen 0.703; best of 8 samples 0.768; 56.4% of greedy-missed values appear in at least one of 8 samples (J, GRPO part 3 table). Size cut structure errors (missing 476 to 213) but not misreads (279 to 258) (J L109-125).

## 1. Knowledge distillation
**Mechanism.** The student matches the teacher's full next-token distribution instead of one-hot labels. Variants: sequence-level (train on teacher outputs) [4]; token-level on fixed text (forward KL) [27]; on-policy (student samples, teacher scores each token, reverse KL or JSD) [1][2].

**Evidence.**
- At pre-training scale: Gemma 2 2B distilled from a 7B teacher averages 67.7 vs 60.3 from scratch at 500B tokens [11]. Llama 3.2 1B/3B were pruned from 8B, then trained on 8B/70B logits [12]. Minitron found forward KL beat reverse KL, MSE and cosine for the logit loss [27].
- On-policy: GKD gives 2.1x / 1.7x / 1.9x relative gains over the initial T5 student on summarisation / translation / GSM8K; best divergence is task-dependent [1]. MiniLLM: dropping teacher-mixed sampling cost 4.2 ROUGE-L on GPT-2 125M, and the paper links it to degenerate repetition [2]. DistiLLM claims up to 4.3x faster training and says GKD/MiniLLM can hurt small students [3]. Qwen3-8B: off-policy distillation 55.0 AIME'24, plus RL 67.6 at 17,920 GPU-h, plus on-policy distillation 74.4 at 1,800 GPU-h [10]; a blog reproduces the pattern at 9-30x lower cost [13] (blog, lead only).
- VLM-specific, all modest: LLaVA-KD, Qwen2.5-0.5B student, TextVQA 49.2 to 52.0, ten-benchmark average 56.5 to 58.9; with the 3B teacher +2.4, with the 7B teacher +1.8, so the bigger teacher gave less; same LLM family required for a shared vocabulary [5]. Align-KD, 1.7B student with 7B teacher: average 62.4 to 64.4 and 63.7 to 65.1, TextVQA +1.2/+0.9 [6]. DistillVLM: VQA +0.8, CIDEr +5.1 [15]. LLaVA-MoD distils 7B into 0.5B/1.8B students on 5M samples [16]. Mini-InternVL distilled only the vision encoder (InternViT-6B to 300M, cosine loss) [8]; the InternVL3 report mentions no distillation of the 1B/2B [9]. Our 300M ViT is already a distilled model.
- Document/OCR: I found no paper that distils a 7B+ VLM into a 0.5B-LM VLM with soft logits on OCR or extraction. Closest are sequence-level: LightOnOCR (1B student on Qwen2-VL-72B / Qwen3-VL-235B transcriptions, 83.2 on OlmOCR-Bench, no logit loss) [25], and DocKD (LLM annotations on unlabelled documents: DocVQA ANLS 80.6 human-only to 83.4 combined) [24].
- Capacity gap: students degrade when the teacher is far larger [14]; LLaVA-KD agrees [5]; Minitron found the opposite (15B teacher beat 8B for a 4B student) [27]. Distillation wins when a teacher already exists [32].

**Targets.** Invalid JSON/loops (6.7% of values): on-policy feedback penalises repeats the student generates. Values left out (10.6%): at the point where the student drifts, the teacher puts mass on the skipped item; consistent with the 56.4% figure above, but I found no paper measuring recall effects. French reading/misreads (2.9%): little, since the 8B did not reduce misreads. Generalisation gap: soft labels on training pages add little (student already 0.91 there); the gain must come from images the student has not memorised, and on-policy distillation needs no labels for them.

**Cost (est.).** Teacher: 8B LoRA, 6 epochs took 41 min (J L109; L98 says 4 H100, L109 says 2 GPUs), about 0.6-1.2 GPU-s per fwd+bwd sample; adapters were deleted (J L573), so retrain: ~1-1.5 GPU-h. Teacher forward only is about a third: +0.5-1 GPU-h over a ~1 GPU-h student run (8,358 samples). On-policy stage: ~1,400-2,000 pages x ~1k generated tokens, under 1 GPU-h with batched generation.
**Effort.** About 40 lines: subclass `Trainer.compute_loss` in `train.py`, run the teacher under `no_grad` on the same batch, KL on `labels != -100` positions. For on-policy, hand-roll with `generate.py` or smoke-test TRL's `DistillationTrainer` on 20 pages.
**Risks.** Teacher errors imprint (258 misreads). The 8B peaked at epoch 2 of 6 (J L109), so a fully trained teacher has near one-hot outputs on training pages. Memory: the 1B run with 2 pages per step peaked at 78.7 GB without checkpointing (J R_ft_batch2); an 8B teacher in bf16 adds ~16 GB, so use 1 page per step or enable checkpointing. Capacity gap: include a 2B teacher control.

## 2. Pseudo-labels and self-training
**Mechanism.** Fine-tune the 8B, write JSON for unlabelled pages, train the 1B on gold plus these, with noise (augmentation, dropout).
**Evidence.** Noisy Student: 300M unlabelled images, 88.4% top-1, noise on the student is essential [20]. Self-training for sequence generation: dropout is critical [21]. Self-training gave +1.3 to +3.4 AP at every labelled-set size where pre-training stopped helping [22]. UniversalNER: 45.9k ChatGPT-labelled passages; 7B/13B students beat ChatGPT by 7-9 F1 zero-shot [23]. DocKD and LightOnOCR above [24][25].
**Against, from this project:** 993 extra labelled pages gave +0.011 (C "Already tried"); GRPO on 974 unseen pages gave 0.708/0.714 and 0.718/0.712 vs start 0.705/0.719 (J L1391). 8B labels (0.82) are noisier than the GLM labels (0.91) the student already trains on. Expect at most +0.01-0.02 unless pages are French and in-domain.
**Where unlabelled images could come from:**
- 557 CVs / 725 pages of the same Roboflow dataset (CC BY 4.0) left out of training because they are only 640 x 640 px (checked in `data/processed/documents.jsonl`; J L183 says ~670). The only licence-clean in-domain pool I know of, but the images are small and squared, so the teacher may read them badly: test 20 pages first.
- Roboflow Universe resume sets (HWRD, Cerebra, Haris; CC BY 4.0 per search snippets) are mostly section-detection sets; counts and languages unverified [39].
- HF `amnamine/ResumeDataset`: 100-166 files, no licence, apparently real applicants: do not use [38].
- RVL-CDIP "resume" class: ~25,000 grayscale scans, max 1,000 px, licence "other", from the tobacco-litigation archive; English, far from French digital CVs [37].
- hadikp (8,905 English PDFs, licence unknown) is already measured: no gain.
- Synthetic: ResumeBench has 2,500 synthetic resumes in 5 languages but release and licence are unconfirmed [40]. Alternative: a text LLM under Apache-2.0 (Qwen2.5-7B/14B-Instruct) writes French CV JSON, a template renders it; labels are exact, so no teacher noise [35].
**Cost (est.).** vLLM serves InternVL3 HF format [36]; 10k pages x 1k tokens = 10M tokens, ~1-2 GPU-h at an assumed 2-3k tok/s per GPU: measure on 20 pages first. **Effort:** low if `generate.py` output is reused. **Risk:** compounding teacher errors, rendered-vs-real domain gap.

## 3. Prune a 2B-4B VLM below 1B, then distil
**Evidence.** Minitron: 15B to 8B/4B with under 3% of the data, up to 40x fewer tokens; pruned+distilled 4B (100B tokens) beat a 150B-token scratch 4B on MMLU 37.8 vs 24.4 [27]. Sheared-LLaMA: 0.4B tokens pruning plus 50B continued training for ~3% of from-scratch compute [28]. For VLMs, a 2026 study pruned LLaVA-7B, Mini-InternVL-4B, Bunny-3B at 15-60%: recovery retained 78-84% of performance at 60%, said extreme pruning "is not a viable approach", and widthwise pruning won in low-data regimes [30].
**Why not here.** Under 1B total with a 0.30B ViT leaves ~0.69B for the LLM. InternVL3-2B's LLM is 1.78B untied, a 61% cut, the regime [30] calls unrecoverable; the 8B would need 91%. The 2B LoRA is only 0.754/0.716, so the pruned model starts from a source barely above our 1B. Recovery budgets are 10^10 tokens [27][28]; our corpus is ~10^7 per epoch (est.). Vocabulary trimming is a separate lever: LightOnOCR cut 151k to 32k tokens, 1005.6M to 883.6M parameters, +11.6% speed, OlmOCR 76.1 to 73.1 [25].

## 4. Merging / upcycling
Weight selection initialises a small model with a subset of a larger same-family model's weights: ViT-T on ImageNet-1K +1.6, Pets +43.6, and it stacks with KD (CIFAR-100: 78.4 KD alone, 81.4 selection, 83.9 both); image classification only, smaller teacher was better [31]. Our shapes differ 4x in width (3,584 to 896, heads 28 to 14, MLP 18,944 to 4,864) [19], and the shipped 1B already has ~200B tokens of multimodal pre-training [9]; replacing it with a slice of the 8B is likely a regression (my judgement). Cross-architecture fusion methods (GenRecal [42]) are distillation again. No evidence found of parameter-space knowledge transfer into a smaller VLM. Seed soups (+0.02, C) already use merging.

## 5. Licences and permissions
- InternVL3 licence metadata conflicts. `InternVL3-8B` says apache-2.0 with a Qwen licence link and README "MIT, Qwen2.5 Apache-2.0" [34]; the `-hf` repos (including our student base) say "other / qwen" and "Qwen License" [34][35]. The linked Qwen licence (§5b) requires "Built with Qwen" or "Improved using Qwen" in documentation when outputs or the materials are used to train a distributed model; I found no ban on training other models, and no non-commercial conflict [33]. Cheap fix: add that line to the model card. Not legal advice.
- Qwen2.5 0.5B/1.5B/7B/14B/32B and Qwen2.5-VL-7B are Apache-2.0; Qwen2.5-VL-3B is research-only; 72B is the Qwen licence [35].
- C requires user approval before any larger model is used as teacher, plus approval for the 8B download.

## 6. First experiment: logit distillation from the 8B (method 1)
1. **Teacher:** InternVL3-8B-hf, LoRA r32 + projector as D_8b_lora (J L98), 3 epochs, merged to bf16. Check 0.80+ on gold-50 and val first.
2. **Student:** `configs/internvl3-1b.toml` unchanged (full fine-tune, 6 epochs, 12 tiles, augment 0.5, vision encoder trained), seeds 1-3 per arm.
3. **Arms:** A baseline. B: 0.5 CE(gold) + 0.5 forward KL(teacher||student), T=1, answer tokens, teacher sees the same augmented image. C: B, then one epoch on-policy (temperature 1, cap 1,600 tokens, JSD beta 0.5 or reverse KL, lr 3e-6) over the 1,393 training pages plus the 725 leftover 640 px Roboflow pages (if the teacher reads them), no labels. D (optional): B with a 2B teacher.
4. **Comparison:** compare against single-seed baselines, not the 10-seed soup (0.732/0.724 includes the +0.02 soup effect). Choose epoch on validation, report gold-50 at that epoch, strict/lenient/content/repaired. Also count unparsable pages (baseline 6.7%) and left-out values (10.6%).
5. **Adopt if** the 3-seed mean beats the baseline mean by more than 0.02 on both sets, then soup the KD seeds against a same-size baseline soup.
6. **Cost (est.):** ~10 student runs x 1.5 GPU-h plus the teacher: 15-20 GPU-h, about 4-5 h wall-clock on 4 GPUs.

## Could not verify
- Roboflow Universe pages returned HTTP 403: image counts, languages and licences of resume sets come from search snippets only.
- ResumeBench release and licence; RVL-CDIP/IIT-CDIP actual terms of use (UCSF page had no retrievable text).
- A "VLM-KD" paper (not found); no source for soft-logit KD of a 7B+ VLM into a 0.5B-LM VLM on OCR or extraction.
- TRL `DistillationTrainer` with InternVL-hf; vLLM with merged-LoRA 8B checkpoints. Untested.
- Gemma 2 Table 6 and LLaVA-KD Table 3 were read through a summarising fetch; the Table 3 row mapping was inferred.
- Blog multipliers (9x/18x/30x) are the blog's own [13].
- Which InternVL3 licence governs: metadata conflicts.
- All GPU-hour and throughput figures are my estimates.

## Sources
[1] https://arxiv.org/abs/2306.13649 (GKD; html https://arxiv.org/html/2306.13649)
[2] https://arxiv.org/abs/2306.08543 (MiniLLM)
[3] https://arxiv.org/abs/2402.03898 (DistiLLM)
[4] https://arxiv.org/abs/1606.07947 (Kim & Rush)
[5] https://arxiv.org/html/2410.16236 (LLaVA-KD)
[6] https://arxiv.org/html/2412.01282 (Align-KD)
[7] https://arxiv.org/abs/2501.01709 (MoVE-KD)
[8] https://arxiv.org/html/2410.16261 (Mini-InternVL)
[9] https://arxiv.org/html/2504.10479 (InternVL3)
[10] https://arxiv.org/html/2505.09388 (Qwen3, Table 21)
[11] https://arxiv.org/html/2408.00118 (Gemma 2)
[12] https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
[13] https://thinkingmachines.ai/blog/on-policy-distillation/
[14] https://arxiv.org/abs/1902.03393 (teacher assistant)
[15] https://arxiv.org/abs/2104.02096 (Compressing visual-linguistic model via KD)
[16] https://arxiv.org/html/2408.15881 (LLaVA-MoD)
[17] https://huggingface.co/docs/trl/main/en/gkd_trainer ; https://github.com/huggingface/trl/blob/main/trl/experimental/gkd/gkd_trainer.py
[18] https://huggingface.co/docs/trl/main/en/distillation_trainer ; https://github.com/huggingface/trl/blob/main/trl/trainer/distillation_trainer.py
[19] https://huggingface.co/OpenGVLab/InternVL3-1B-hf/raw/main/config.json ; .../InternVL3-8B-hf/raw/main/config.json ; .../InternVL3-14B-hf/... ; .../InternVL3-38B-hf/...
[20] https://arxiv.org/abs/1911.04252
[21] https://arxiv.org/abs/1909.13788
[22] https://arxiv.org/abs/2006.06882
[23] https://arxiv.org/html/2308.03279
[24] https://arxiv.org/html/2410.03061
[25] https://arxiv.org/html/2601.14251
[27] https://arxiv.org/html/2407.14679 ; https://arxiv.org/abs/2408.11796
[28] https://arxiv.org/html/2310.06694
[30] https://arxiv.org/html/2604.24380
[31] https://arxiv.org/html/2311.18823
[32] https://arxiv.org/abs/2502.08606
[33] https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/LICENSE
[34] https://huggingface.co/OpenGVLab/InternVL3-8B ; https://huggingface.co/OpenGVLab/InternVL3-8B-hf
[35] https://huggingface.co/api/models/<model> for each Qwen2.5 / InternVL3 repo named
[36] https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md
[37] https://huggingface.co/datasets/aharley/rvl_cdip
[38] https://huggingface.co/datasets/amnamine/ResumeDataset
[39] https://universe.roboflow.com/hwrd/resumes-4xmu5 ; https://universe.roboflow.com/cerebra/resume-nuhis ; https://universe.roboflow.com/haris-workspace-0ndy3/resume-section
[40] https://aclanthology.org/2025.emnlp-main.1626/
[42] https://arxiv.org/abs/2506.15681 (GenRecal)
