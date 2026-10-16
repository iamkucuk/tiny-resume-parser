# Development log — InternVL3-1B LoRA, CV page image → ATS JSON

One entry per version. Each answers: what we ran into, what we observed, what we tried, what worked and what did not.
Full result tables: `DESIGN_NOTES.md` Part 3. Research behind the choices: `literature/`.
Scores are on the first 50 reviewed gold pages (`labels_gold_v6_verified.jsonl`, CVs never trained on) unless stated.
Field F1 = exact match of (field path, value) pairs; unparsed answers count 0.

---

## Smoke tests (Mac, 2026-10-11)
- **Ran into:** first gate (50 pages, 26 steps) gave 0/10 valid JSON; MPS pure-bf16 runs had grad-norm spikes up to ~1e14.
- **Observed:** no pipeline bug — the model wrote generic JSON-Resume keys and repeated entries. An 8-epoch run looped inside `corrections` on 5/10 pages.
- **Tried:** longer smoke run; mixed precision on CUDA (fp32 LoRA weights, bf16 compute).
- **Worked:** mixed precision removed the spikes. **Did not:** we read the 8-epoch loops as "underfit" without locating where answers broke — they were the first sign of the `corrections` loop.

## Colab setup (2026-10-11)
- **Ran into:** Colab CPU smoke run stalled at step 0 for 30+ min; peft 0.21 ImportError with Colab's torchao 0.10; the Drive desktop app stopped syncing.
- **Observed:** bf16 matrix multiply on Colab's CPU: 0.2 GFLOP/s vs 18 GFLOP/s in fp32.
- **Worked:** fp32 on CPU-only machines; uninstall torchao in the notebook; rclone (`gdrive:`) for Drive files instead of the desktop app.

## lora_v1 — 2 epochs, full teacher answer (L4)
- **Setup:** 1,393 pages, LoRA r16 on the language model, lr 1e-4, 2 epochs (176 steps, 40 min). Loss 0.80 → 0.22.
- **Observed:** 0/10 valid on training pages; all 10 hit the 2,500-token limit; 8/10 repeated one `corrections` entry (`original` = `corrected`, which never occurs in the labels).
- **Tried:** `no_repeat_ngram_size` 20 → 3/10 valid but syntax slips; 10 → 0/10 (blocks normal JSON).
- **Did not work:** n-gram bans. **Lesson:** low loss did not show the loops; only free generation did.

## lora_v2 — `corrections` removed + Nougat token perturbation
- **Changed:** `corrections` (teacher's spelling-fix list, not visible on the page) left out of the answer; 0.1-probability random token swaps in the input (labels stay correct). Added `src/evaluate.py` (held-out gold, field F1, free vs constrained decoding).
- **Observed:** free 21/50 valid, 14 not JSON, 12 hit the limit, F1 0.43. Constrained 44/50 valid, F1 0.42. Loops now spread over many fields (`technologies`, `summary`, …).
- **Per-field breakdown:** extra `work` entries on 16/36 pages (school, club, project entries put in `work`); misread characters (`OACA` → `OCAAC`), worth ~0.08 F1; teacher spelling fixes explain only 14/285 near misses.
- **Worked:** loops in `corrections` gone; constrained decoding removes all schema errors. **Did not:** constrained decoding does not raise F1.

## lora_v3 — 4 epochs + image-to-text projector trained
- **Changed (two things at once):** 4 epochs (352 steps, 81 min); projector (4.5 M params) trained instead of frozen. Loss 0.74 → 0.11.
- **Observed:** free 35/50 valid, 3 hit the limit, F1 0.58. Constrained 48/50 valid, F1 0.58. Extra `work` entries 16/36 → 8/44 pages. Every section up (`work` 0.47 → 0.62, `awards` 0.07 → 0.30); `extra` still 0.
- **Worked:** clear gain on all measures. **Unknown:** which change caused it. The near-miss gap (similarity F1 − exact F1) stayed ~0.08–0.09, so the projector shows no clear effect on misreads.

## Between v3 and v4 (2026-10-12)
- **Framework research** (`literature/2026-10-12_frameworks_vs_our_recipe.md`): only LLaMA-Factory and TRL train our exact model; our recipe matches their defaults; the gradient-accumulation loss bug does not affect transformers 4.57.6 (checked in installed code). Gaps: per-epoch checkpoints + selection, 12 image tiles instead of 6 (affects 178/1,472 pages), per-component learning rates.
- **Training-label check:** school/club-like entries in `work` — 2.8% of training labels vs 2.9% of reviewed gold labels. Wrong-section errors come from the model, not from the labels.
- **Speed test (L4, same 16 pages per step):** checkpointing on 0.85 s/page, 8.9 GB; **off 0.67 s/page (1.27×), 14.6 GB**; 4 or 8 pages per pass without checkpointing ran out of memory. The model has no BatchNorm (LayerNorm/RMSNorm only), so batch size changes speed, not results.
- **Scoring speed:** `evaluate.py --batch N` generates N pages together; answers identical to one-at-a-time on 3 pages (free and constrained). GPU speed-up measured in lora_v4 (3.2×).

## lora_v4 — 8 epochs, adapter saved per epoch
- **Changed (one thing vs v3):** 8 epochs (704 steps, 2 h 19 min, 11.9 s/step, 16.2 GB); adapter saved after every epoch; gradient checkpointing off (speed only). Mean loss per epoch 0.33 → 0.06; largest grad norm 1.97.
- **Question:** does longer training still help, or does it start to overfit? Epochs 4, 6, 8 scored from one run.
- **Observed:** F1 peaks at epoch 6, then falls. Free F1: ep4 0.57, **ep6 0.64**, ep8 0.60. Constrained: ep6 **50/50 valid, F1 0.63**; ep8 49/50, F1 0.60. Bootstrap over the 50 pages (2,000 resamples): ep6 > ep8 in 99%, ep6 > v3 in 100%, ep8 > v3 in 77%.
- **Scoring speed:** 8 pages per batch: 8.6 s/page vs 27.7 s one at a time (3.2×).
- **Worked:** per-epoch checkpoints found a better adapter than the last one. **Did not:** training past ~6 epochs (loss still falls, gold F1 drops). Free ep6 has 12 schema errors; constrained decoding removes them at a cost of 0.01 F1.
- **Caveat:** one training run, 50 pages; the bootstrap covers page-to-page variation, not run-to-run variation.
- **Use now:** `runs/lora_v4/checkpoint-528` (epoch 6).

## lora_v5 — lora_v4 with the projector frozen
- **Changed (one thing vs v4):** image-to-text projector frozen (LoRA on the language model only, 8.8 M trainable vs 13.3 M; adapter 35 MB vs 53 MB). Same 8 epochs, seed 42, data order and perturbation. 2 h 14 min; mean loss per epoch 0.35 → 0.08 (v4: 0.33 → 0.06); largest grad norm 2.03.
- **Question:** does training the projector help? v3 changed epochs and projector together.
- **Observed (F1, frozen vs trained):** ep6 free 0.61 vs 0.64, ep6 constrained 0.62 vs 0.63, ep8 free 0.60 vs 0.60, ep8 constrained 0.58 vs 0.60. Paired bootstrap, frozen − trained: −0.027 (95% range −0.075 to +0.013), −0.010, −0.001, −0.027 (−0.070 to +0.011). Frozen gave more valid free answers at ep6 (41 vs 34).
- **Conclusion:** training the projector helps by 0–0.03 F1; all four comparisons point the same way or are equal, none is clearly outside page-to-page noise. It costs no measurable time, so keep it on. v3's gain over v2 came mainly from more epochs.
- **Caveat:** one run per setting, 50 pages.

## lora_v6 — 12 image tiles + LoRA rank 32
- **Changed (two things vs v4, user's choice):** tile cap 6 → 12 (`--max-patches`, new flag in `train_lora.py` and `evaluate.py`; scoring must use the same cap); LoRA rank 16 → 32 (alpha 64). Projector trained, 8 epochs, seed 42.
- **Measured before the run:** 12 tiles change 156/1,393 training pages (7 → 13 tiles) and only 6 of the 50 scored gold pages (27/254 in all gold). Longest sample 3,855 → 4,778 tokens; memory estimated ~20 GB of 24 without gradient checkpointing.
- **How to separate the two changes:** score the 6 tall gold pages apart from the other 44 (those see the same image input as v4, so only rank and training differ).
- **Run:** 2 h 31 min; mean loss per epoch 0.30 → 0.036; largest grad norm 2.27 (epoch 1 only).
- **Observed (F1, v6 vs v4):** ep6 free 0.61 vs 0.64, ep6 constrained 0.62 vs 0.63, **ep8 free 0.63 vs 0.60, ep8 constrained 0.64 vs 0.60** (49/50 valid). Paired bootstrap v6 − v4: ep8 free +0.032 (v6 higher in 84%), ep8 constrained +0.039 (95% range +0.004 to +0.075, 98%), ep6 constrained −0.004 (45%). Free ep6: 6 tall pages +0.094 (96%), other 44 −0.048.
- **Conclusion:** best epoch moves from 6 to 8 (rank 32 fits more slowly, loss still falling). v6 ep8 constrained (0.64) ties v4 ep6 free as the best score and is the best constrained score; the 12-tile gain is on tall pages only. Rank and tiles were changed together, so the gain is not split between them.
- **Caveat:** one run, 50 pages; base model before training still not scored (cells exist in the notebook, never run).

## Before DPO — failure rate on training pages + base model (2026-10-12)
- **lora_v6 final adapter, 120 training pages, free decoding (L4, 15 min, ~7 s/page):** 112 valid, 5 schema errors (wrong enum values like `projects.type: "internal"`, keys put under the wrong section), 3 not JSON (1 repetition loop to the token limit, 2 syntax slips). Field F1 0.85 (median page 0.88); 27 pages below 0.80. So the model fails hard on ~7% of pages it was trained on and is imperfect on ~1/4.
- **Base model before training, 50 gold pages:** 0/50 parsed, F1 0. It wraps answers in ```json fences and invents its own keys (`level`, `institution`, `profil_personnel`), because our prompt gives no schema. This only shows the base model does not know our format; a fair zero-shot baseline would need the schema in the prompt.
- **DPO data estimate:** on all 1,346 training pages ~90 hard failures + ~300 low-F1 answers as rejected answers (label = chosen); ~2.5 h of L4 generation.

## TRUBA runs, 2026-10-13 (H100/H200, one GPU each unless noted)
All: InternVL3-1B-hf, 12 tiles, 1,393 training pages, scored at each epoch on the dev pages (gold 1–50 and the
79 validation pages), scoring 8 pages per batch, free decoding. Scores are lenient F1 (`rescore.py`); "canonical" =
`rescore.py --canon`. Full fine-tunes: all language-model and projector weights, vision encoder frozen unless noted.
Hypotheses and reasoning: `RESEARCH_JOURNAL.md`.

| Run | One change vs reference | Train time | Gold-50 by epoch | Validation by epoch | Verdict |
|---|---|---|---|---|---|
| A_v6_seed1 | lora_v6 recipe, seed 1 | 47 min | ep4–8: .640 .661 .623 .656 .662 | .624 .650 .653 .651 .643 | seed spread ~0.01–0.02 for LoRA |
| **B_fullft_1b** | full fine-tune, lr 1e-5, 6 ep | 26 min | ep2–6: — .668 .683 **.693** .689 | — — — **.669** — | +0.03 over LoRA; new baseline |
| I_ft_seed1 | B, seed 1 | 35 min | ep3–6: .660 .686 .686 .688 | .636 .660 **.677** .665 | B's spread < 0.01 at best epoch |
| J_ft_lr2e5 | B, lr 2e-5 | 37 min | ep3–6: .638 .661 .693 .693 | .639 .660 .668 **.688** | ties B; slower start |
| K_ft_canon | B, canonical training labels (`src/canon.py`) | 32 min | ep3–6: .675 .683 .672 .680 | .630 .641 .658 .667 | rejected: −0.01 raw, also behind scored canonically |
| E_dpo_v6 | DPO from lora_v6 on its own failures | 39 min job | step 30 .583, 60 .556, end .539 | .584 .569 .548 | rejected: every step lowers F1 (lora_v6 .650) |

Cancelled at 13:38 (daytime job limit on the shared cluster); scores up to that point:
| Run | Change vs B | Gold-50 | Validation | Verdict |
|---|---|---|---|---|
| G_ft_24tiles | 24 tiles (train 51 min on H200) | ep3–5: .637 .641 .645 | .640 .629 .674 | gold −0.05; validation ep5 ties B → rejected for now |
| H_ft_lr5e6_8ep | lr 5e-6, 8 epochs | ep3–7: .614 … .653 .640 .644 | best .634 (ep4), ep6 .626 | rejected: slower and lower |
| N_ft_canon_seed1 | K with seed 1 | ep4–6: .685 .669 .665 | ep4 .652, ep5 .628 | confirms K is rejected |
| M_ft_combo | canonical labels + 24 tiles + vision encoder | not scored (cancelled during epoch 6) | — | dropped: two of its three parts already rejected |
| L_ft_vision | vision encoder trained too | ep4–6: .681 .700 .679 | ep4 .661, ep5 .643 | gold ep5 best of all 1B runs, but validation is lower → not a gain yet |

- **Out of scope (larger than 1B; kept only as measurements):** C 2B LoRA best gold .754 (ep7), validation .716;
  D 8B LoRA (2 GPUs) gold .814 (ep5), validation .790 (ep3); F 8B with 24 tiles gold .824 (ep5), validation .812 (ep6).
  Size halves structure errors but not misreads.
- **New flag:** `train_lora.py --augment P` (random resolution/rotation/contrast/blur/JPEG noise on training page images with probability P).
- **Kept on disk:** best-epoch checkpoint + `model/` per run; other checkpoints deleted (3.5 GB each).
- **Scoring batch 32 vs 8 (B ep5 gold):** 0.685 vs 0.693, 18/50 answers identical, 1.6× faster → keep 8 for comparability.
- **Constrained decoding, B ep5 gold (P_constr_B5):** gold 0.694 vs 0.693 free, validation 0.676 vs 0.669, 49/50 valid, 5.4× slower → not an F1 lever; for deployment only.
- **O_ft_half (B on the first 696 pages, 6 epochs):** gold ep3–6 .613 .610 .621 .641, validation ep3–6 .572 .578 .592 .584; half the steps too, so repeat with 12 epochs (O2).
- **R_ft_batch2 (2 pages per step):** gold ep4–6 .686 .669 .664, validation .664 .653 .651 → ~0.02 below B at best; rejected, keep 1 page per step.
- **U_ft_aug50 (B + image noise on 50% of pages):** gold ep4–6 .691 .695 **.727**, validation .631 .668 .680 → best gold of any 1B run; validation inside noise; seed repeat pending.
- **Still running:** P_constr_B5 (B ep5 with constrained
  decoding), S_eval_b32 (B ep5 scored 32 pages per batch, speed test).

## Open items
- ~~Projector on/off test~~ done in lora_v5: keep it trained.
- ~~12 image tiles for tall pages~~ done in lora_v6: helps tall pages; best constrained score so far.
- Score the base model before training (zero-shot JSON baseline).
- Separate, lower learning rate for the projector.
- Final score on all 254 gold pages before moving to SmolVLM-500M.

### Q_ft_vision_seed1 (TRUBA 1602110, 57 min) — full FT + vision encoder trained, seed 1
- Args: `--full-ft --lr 1e-5 --epochs 6 --max-patches 12 --no-grad-ckpt --train-vision --seed 1`.
- Lenient F1 ep4–6: gold 0.695 0.725 0.714; val 0.660 0.683 0.701. Val-chosen ep6: gold 0.714 / val 0.701
  (same-seed baseline I: 0.686 / 0.677). Seed-42 pair (L vs B) was mixed; third seed AA_vision_seed2 queued.

### W_ft_aug100 (TRUBA 1602130, 53 min) — full FT + image noise on every page (--augment 1.0)
- Lenient F1 ep4–6: gold 0.672 0.717 0.705; val 0.651 0.668 0.680. Val-chosen ep6: gold 0.705 / val 0.680.
- Same as --augment 0.5 (U, ep6: 0.727 / 0.680) on val, lower on gold → keep P 0.5.

### V_ft_aug50_8ep (TRUBA 1602129, 64 min) — full FT + image noise P 0.5, 8 epochs
- Lenient F1 ep5–8: gold 0.711 0.700 0.715 0.717; val 0.657 0.678 0.677 0.675. Val-chosen ep6: gold 0.700 / val 0.678.
- No gain from 8 epochs over 6 (U ep6: 0.727 / 0.680). Keep 6 epochs.

### Y_aug_seed2 (TRUBA 1602169, 50 min) — full FT + image noise P 0.5, seed 2
- Lenient F1 ep4–6: gold 0.674 0.694 0.711; val 0.659 0.667 0.674. Val-chosen ep6: gold 0.711 / val 0.674.
- Same-seed reference AB_base_seed2 still running.

### AB_base_seed2 (TRUBA 1602219, 54 min) — baseline full FT, seed 2 (same-seed reference for Y and AA)
- Lenient F1 ep3–6: gold 0.633 0.674 0.686 0.666; val 0.632 0.666 0.659 0.655. Val-chosen ep4: gold 0.674 / val 0.666.

### AA_vision_seed2 (TRUBA 1602207, 62 min) — full FT + vision encoder trained, seed 2
- Lenient F1 ep3–6: gold 0.609 0.682 0.694 0.689; val 0.628 0.691 0.693 0.680. Val-chosen ep5: gold 0.694 / val 0.693
  (same-seed baseline AB: 0.674 / 0.666).

### X_aug_lr2e5 (TRUBA 1602237, 55 min) — full FT + image noise P 0.5, lr 2e-5 (one change vs U: lr 1e-5 → 2e-5)
- Lenient F1 ep3–6: gold 0.699 0.712 0.705 0.710; val 0.643 0.684 0.717 0.714. Val-chosen ep5: gold 0.705 / val 0.717
  (U ep6: 0.727 / 0.680). Highest val so far. Seeds 1 and 2 queued (X1, X2).

### Z_aug_vision (TRUBA 1602272, 62 min) — full FT + image noise P 0.5 + vision encoder trained (combined), seed 42
- Lenient F1 ep3–6: gold 0.686 0.661 0.699 0.705; val 0.680 0.671 0.720 0.719. Val-chosen ep5: gold 0.699 / val 0.720.
- vs B (0.693 / 0.669): +0.006 gold, +0.051 val. Highest val so far; gold gain of image noise alone (U 0.727) not seen here.

### Z1_aug_vision_seed1 (TRUBA 1602415, 64 min) — combined (image noise P 0.5 + vision encoder), seed 1
- Lenient F1 ep3–6: gold 0.681 0.666 0.705 0.722; val 0.685 0.688 0.699 0.711. Val-chosen ep6: gold 0.722 / val 0.711
  (Z seed 42 ep5: 0.699 / 0.720).

### X1_aug_lr2e5_seed1 (TRUBA 1602420, 59 min) — image noise P 0.5, lr 2e-5, seed 1
- Lenient F1 ep3–6: gold 0.620 0.716 0.702 0.714; val 0.605 0.671 0.676 0.695. Val-chosen ep6: gold 0.714 / val 0.695
  (X seed 42 ep5: 0.705 / 0.717).

### Z2_aug_vision_seed2 (TRUBA 1602416, 65 min) — combined (image noise P 0.5 + vision encoder), seed 2
- Lenient F1 ep3–6: gold 0.641 0.703 0.685 0.697; val 0.661 0.686 0.681 0.693. Val-chosen ep6: gold 0.697 / val 0.693.

### ZX_aug_vision_lr2e5 (TRUBA 1602435, 69 min) — combined (image noise + vision encoder), lr 2e-5 (one change vs Z)
- Lenient F1 ep3–6: gold 0.654 0.677 0.704 0.702; val 0.627 0.660 0.705 0.702. Val-chosen ep5: gold 0.704 / val 0.705
  (Z: 0.699 / 0.720). No gain from the higher lr.

### ZH_hk241 (TRUBA 1602512, 68 min) — combined default + 241 extra hadikp pages (one change vs Z: training file v6_hk)
- Lenient F1 ep3–6: gold 0.668 0.716 0.697 0.702; val 0.639 0.670 0.720 0.712. Val-chosen ep5: gold 0.697 / val 0.720
  (Z: 0.699 / 0.720). Seeds 1 and 2 running / queued.

### X2_aug_lr2e5_seed2 (TRUBA 1602616, 59 min) — image noise P 0.5, lr 2e-5, seed 2
- Lenient F1 ep3–6: gold 0.681 0.703 0.726 0.713; val 0.646 0.691 0.684 0.699. Val-chosen ep6: gold 0.713 / val 0.699.

### ZH1_hk241_seed1 (TRUBA 1602571, 73 min) — combined default + 241 extra hadikp pages, seed 1
- Lenient F1 ep3–6: gold 0.631 0.682 0.722 0.725; val 0.660 0.698 0.701 0.710. Val-chosen ep6: gold 0.725 / val 0.710
  (Z1 same seed without the extra pages: 0.722 / 0.711).

### ZH2_hk241_seed2 (TRUBA 1602701, 70 min) — combined default + 241 extra hadikp pages, seed 2
- Lenient F1 ep3–6: gold 0.671 0.676 0.695 0.722; val 0.640 0.680 0.694 0.710. Val-chosen ep6: gold 0.722 / val 0.710
  (Z2 same seed without the extra pages: 0.697 / 0.693).

### G1_grpo_lr1e-6 (TRUBA 1602542, 152 min) — GRPO from Z checkpoint-528 (ep6), 1 pass over 1,393 pages, lr 1e-6, beta 0
- Lenient F1: gold 0.711 / val 0.718 (start Z ep6: 0.705 / 0.719). Training reward of samples 0.73 → ~0.77.
- Gold: unparsed 3 → 1, token-limit hits 2 → 0. Val: unparsed 2 → 3.

### Z3_aug_vision_seed3 (TRUBA 1602713, 62 min) — combined default, seed 3
- Lenient F1 ep3–6: gold 0.623 0.704 0.699 0.708; val 0.672 0.683 0.708 0.694. Val-chosen ep5: gold 0.699 / val 0.708.

### G1_grpo_lr3e-6 (TRUBA 1602704, 2 h 30 min) — GRPO from Z ep6 (checkpoint-528), lr 3e-6
- Final: gold 0.718 / val 0.704 (start 0.705 / 0.719). Training reward 0.73 → 0.80 by step 150, then flat ~0.78.

### G2_grpo_temp0.7 (TRUBA 1602711, 2 h 29 min) — GRPO from Z ep6, lr 1e-6, sampling temperature 0.7 (was 1.0)
- Final: gold 0.705 / val 0.714 (start 0.705 / 0.719). Training reward ~0.85 (higher because samples are closer to greedy).

### S_soup4_default (TRUBA 1602956, 9 min) — weight average of Z ep6, Z1 ep6, Z2 ep4, Z3 ep5 (no training)
- gold 0.721 (P 0.758 R 0.688) / val 0.711. Members' mean 0.707 / 0.706; best member 0.722 / 0.711.

### S_soup10_vision (TRUBA 1602963, 9 min) — weight average of 10 vision-trained full fine-tunes (no training)
- gold 0.728 / val 0.732. Default 4-seed mean 0.704 / 0.708 → +0.024 / +0.024; highest val of any 1B model so far.

### S_soup16_wide (TRUBA 1602964, 14 min) — S_soup10_vision members + 6 image-noise-only (vision frozen) models
- gold 0.744 (P 0.757 R 0.732) / val 0.723. vs 10-model average 0.728 / 0.732: gold +0.016, val −0.009.

### G3_grpo_beta0.04 (TRUBA 1602782, 2 h 39 min) — GRPO from Z ep6, lr 1e-6, KL penalty beta 0.04 (was 0)
- Final: gold 0.707 / val 0.715 (start 0.705 / 0.719). No change. Model deleted (GRPO not adopted).

### Z4_aug_vision_seed4 / Z5_aug_vision_seed5 (TRUBA 1602988/89, 62 min each) — default recipe, seeds 4 and 5
- Z4 lenient ep3–6: gold 0.655 0.704 0.664 0.711; val 0.690 0.688 0.714 0.714. Val-chosen ep5 (tie with ep6, first kept): gold 0.664 / val 0.714.
- Z5 lenient ep3–6: gold 0.666 0.686 0.710 0.689; val 0.667 0.694 0.677 0.698. Val-chosen ep6: gold 0.689 / val 0.698.

### Z6_aug_vision_seed6 (TRUBA 1602990, 63 min) — default recipe, seed 6
- Lenient ep3–6: gold 0.649 0.706 0.738 0.737; val 0.660 0.680 0.701 0.712. Val-chosen ep6: gold 0.737 / val 0.712.

### S_soup13_vision / S_soup19_wide (TRUBA 1603094/95, 13–15 min) — averages with Z4, Z5, Z6 added
- 13 vision-trained: gold 0.725 / val 0.730 (10-model: 0.728 / 0.732). 19 wide: gold 0.745 / val 0.724 (16-model: 0.744 / 0.723).

### SV_smolvlm_default (TRUBA 1602995, 87 min) — SmolVLM-500M, InternVL default recipe (full FT + vision, noise 0.5, lr 1e-5, 6 ep)
- Lenient ep3–6: gold 0.300 0.396 0.374 0.370; val 0.345 0.418 0.437 0.460. Val-chosen ep6: gold 0.370 / val 0.460.
- ep6 gold: 34 valid, 5 schema errors, 11 unparsed (10 hit the 3,000-token limit — all repetition loops); val 14 unparsed.

### Z7_aug_vision_seed7 (TRUBA 1603109, 66 min) — default recipe, seed 7
- Lenient ep3–6: gold 0.665 0.709 0.681 0.701; val 0.648 0.689 0.696 0.711. Val-chosen ep6: gold 0.701 / val 0.711.

### Z8_aug_vision_seed8 (TRUBA 1603110, 78 min) — default recipe, seed 8
- Lenient ep3–6: gold 0.672 0.701 0.722 0.736; val 0.677 0.697 0.707 0.708. Val-chosen ep6: gold 0.736 / val 0.708.
- Default recipe over 9 seeds (val-chosen): mean gold 0.705 / val 0.708.

## SV_reppen1.05 / SV_reppen1.1 (2026-10-14) — scoring only
- SV_smolvlm_default ep6 weights, `--repetition-penalty 1.05 / 1.1`. Lenient F1 gold/val: 0.342/0.430 and 0.370/0.407
  (baseline 0.370/0.460). Rejected.

## SV_lr3e-5 (2026-10-14)
- SmolVLM-500M full FT, default recipe with lr 3e-5. Lenient gold/val ep3–6: 0.431/0.436, 0.519/0.508, 0.513/0.542,
  0.536/0.521. Val-chosen ep5 0.513/0.542 (baseline lr 1e-5: 0.370/0.460). Kept checkpoint-440 (ep5).

## SV_tiles30 (2026-10-14)
- SmolVLM, lr 1e-5, --max-patches 30. Lenient gold/val ep3–6: 0.306/0.388, 0.428/0.456, 0.399/0.439, 0.423/0.448.
  Val-chosen ep4 0.428/0.456 (baseline 0.370/0.460). Not adopted; weights deleted.

## SV_ep10 (2026-10-14)
- SmolVLM, lr 1e-5, 10 epochs. Lenient gold/val ep6–10: 0.424/0.443, 0.420/0.465, 0.416/0.492, 0.462/0.487, 0.419/0.485.
  Val-chosen ep8 0.416/0.492 (baseline 0.370/0.460; lr 3e-5 run 0.513/0.542). Weights deleted.

## ZK_hk1k_seed42 / seed1 / seed2 (2026-10-14)
- InternVL default recipe + `labels_train_v6_hk1k.jsonl` (2,465 pages). Val-chosen lenient gold/val: 0.723/0.722 (ep6),
  0.729/0.713 (ep5), 0.734/0.740 (ep6); mean 0.729/0.725 vs 9-seed default 0.705/0.708. Adopted.

## SV_lr6e-5 (2026-10-14)
- SmolVLM, lr 6e-5, 6 epochs. Lenient gold/val ep3–6: 0.487/0.503, 0.551/0.566, 0.550/0.558, 0.545/0.589. Val-chosen ep6
  0.545/0.589 (lr 3e-5: 0.513/0.542). Adopted.

## SV_hk1k_lr3e-5 (2026-10-14)
- SmolVLM, lr 3e-5, `labels_train_v6_hk1k.jsonl`. Lenient gold/val ep3–6: 0.446/0.489, 0.457/0.456, 0.495/0.518, 0.519/0.501.
  Val-chosen ep5 0.495/0.518 (SV_lr3e-5: 0.513/0.542). Not adopted; weights deleted.

## SV_lr6e-5_ep10 (2026-10-14)
- SmolVLM-500M full FT + vision, lr 6e-5, 10 epochs, 12 tiles, augment 0.5, seed 42. Job 1603745, 1 h 46 min.
- Val-chosen ep9: lenient gold 0.596 / val 0.627; content 0.768 / 0.750. New SmolVLM best (old SV_lr6e-5 0.545 / 0.589).

## GD_lr6e-5 (2026-10-14)
- granite-docling-258M full FT + vision, lr 6e-5, 6 epochs, 12 tiles, augment 0.5, seed 42. Job 1603751, 1 h 23 min.
- Val-chosen ep5: lenient gold 0.542 / val 0.546; content 0.676 / 0.690. 11/79 val answers loop to the token limit.

## GD_lr6e-5_ep10 (2026-10-14)
- granite-docling-258M, same as GD_lr6e-5 but 10 epochs (one change). Job 1604161, 1 h 43 min; scored ep6–10.
- Lenient gold / val: ep6 0.528 / 0.565, ep7 0.605 / 0.570, ep8 0.603 / 0.576, ep9 0.574 / 0.581, ep10 0.590 / 0.579.
- Val-chosen ep9: 0.574 / 0.581 (vs GD_lr6e-5 0.542 / 0.546). Ep9 answers: gold 37 valid / 9 schema error / 4 unparsed;
  val 58 / 16 / 5 (GD_lr6e-5 ep5 val: 11 looped to the limit).

## GD_lr1e-4_ep10 (2026-10-14)
- granite-docling-258M, same as GD_lr6e-5_ep10 but lr 1e-4 (one change). Job 1604826, 1 h 38 min.
- Lenient gold / val: ep6 0.551 / 0.559, ep7 0.591 / 0.576, ep8 0.576 / 0.586, ep9 0.610 / 0.592, ep10 0.609 / 0.596.
- Val-chosen ep10: 0.609 / 0.596 vs GD_lr6e-5_ep10 ep9 0.574 / 0.581 (+0.035 / +0.015). Mean ep6–10 0.587 / 0.582 vs 0.580 / 0.574.

## SV_lr1e-4_ep10 (2026-10-14)
- SmolVLM-500M, same as SV_lr6e-5_ep10 but lr 1e-4 (one change). Job 1604886, 1 h 41 min.
- Lenient gold / val: ep6 0.609 / 0.568, ep7 0.578 / 0.605, ep8 0.612 / 0.614, ep9 0.612 / 0.624, ep10 0.606 / 0.618.
- Val-chosen ep9: 0.612 / 0.624 vs SV_lr6e-5_ep10 0.596 / 0.627 (+0.016 / −0.003). Not adopted; weights deleted.

## ZK_hk1k_seed3 / seed4 / seed5 (2026-10-14)
- InternVL3-1B default recipe + 993 extra hadikp pages (labels_train_v6_hk1k.jsonl), seeds 3/4/5. Jobs 1603737–9, ~1.5 h each.
- Val-chosen (lenient gold / val): seed3 ep5 0.703 / 0.712; seed4 ep5 0.678 / 0.715; seed5 ep6 0.729 / 0.724.
- Averages submitted: S_soup6k (the 6 ZK seeds, 1605070), S_soup16k (S_soup10_vision members + 6 ZK seeds, 1605071).

## GD_lr1e-4_ep10_seed1 / GD_lr6e-5_ep10_seed1 (2026-10-14)
- Seed-1 repeats of both granite learning rates (jobs 1604965/6, ~1 h 36 min each).
- lr 1e-4 seed1, lenient gold / val: ep6 0.568 / 0.587, ep7 0.564 / 0.583, ep8 0.609 / 0.600, ep9 0.622 / 0.635, ep10 0.635 / 0.619 → val-chosen ep9 0.622 / 0.635.
- lr 6e-5 seed1: ep6 0.546 / 0.533, ep7 0.569 / 0.583, ep8 0.610 / 0.589, ep9 0.598 / 0.571, ep10 0.606 / 0.583 → val-chosen ep8 0.610 / 0.589.
- Weights of both lr 6e-5 runs deleted. Submitted: GD_soup2_lr1e-4 (1605090), GD_lr1e-4_ep10_seed2 / seed3 (1605091/2).

## SV_lr6e-5_tiles20 (2026-10-14)
- SmolVLM-500M, SV_lr6e-5_ep10 recipe with 20 tiles (longest_edge 2560 px, 1,344 image tokens) — one change. Job 1604889, 1 h 44 min.
- Lenient gold / val: ep6 0.614 / 0.641, ep7 0.620 / 0.627, ep8 0.634 / 0.645, ep9 0.620 / 0.644, ep10 0.641 / 0.638.
- Val-chosen ep8 0.634 / 0.645 vs 0.596 / 0.627 (+0.038 / +0.018). Mean ep6–10 0.626 / 0.639 vs 0.597 / 0.603 (+0.029 / +0.036).
- Seed repeats submitted: SV_lr6e-5_tiles20_seed1 and SV_lr6e-5_tiles12_seed1 (12-tile baseline, seed 1).

## SV_lr6e-5_tiles30 (2026-10-14)
- SmolVLM-500M, SV_lr6e-5_ep10 recipe with 30 tiles (longest_edge 3072 px, 1,984 image tokens) — one change. Job 1604890, 1 h 53 min.
- Lenient gold / val: ep6 0.635 / 0.608, ep7 0.645 / 0.633, ep8 0.650 / 0.643, ep9 0.646 / 0.639, ep10 0.645 / 0.656.
- Val-chosen ep10 0.645 / 0.656 vs 0.596 / 0.627 (+0.049 / +0.029). Mean ep6–10 0.644 / 0.636 vs 0.597 / 0.603 (+0.047 / +0.033). New SmolVLM best.

## RD_sv → SV_read_ep10 (2026-10-14)
- RD_sv (1604969, 15 min): SmolVLM-500M, 2 epochs of "Write out all text on this page." on 1,976 granite-docling transcripts; loss 1.04 → ~0.3–0.5.
- SV_read_ep10 (1604970, 1 h 43 min): SV_lr6e-5_ep10 recipe started from RD_sv/model (one change). First JSON-training loss 0.86 vs 1.17 from the base model.
- Lenient gold / val: ep6 0.578 / 0.600, ep7 0.578 / 0.589, ep8 0.607 / 0.583, ep9 0.586 / 0.599, ep10 0.601 / 0.604.
- Val-chosen ep10 0.601 / 0.604 vs 0.596 / 0.627 (+0.005 / −0.023). Mean ep6–10 0.590 / 0.595 vs 0.597 / 0.603. Not adopted; weights deleted.

## SV_lr6e-5_tiles12_seed1 / _tiles20_seed1 / _tiles30_seed1 (2026-10-15, jobs 1605094/95/96)
- Seed-1 repeats of the SmolVLM tile-count runs (lr 6e-5, 10 epochs, full FT + vision, augment 0.5).
- Val-chosen lenient gold / val: 12 tiles ep8 0.550 / 0.617; 20 tiles ep7 0.610 / 0.650; 30 tiles ep6 0.587 / 0.651.
- 2-seed means: 12 0.573 / 0.622, 20 0.622 / 0.648, 30 0.616 / 0.654. 30 stays adopted; 20 = 30 within spread.
- Weights of 12- and 20-tile runs deleted; 30-tile seeds kept for SV_soup2_tiles30 (1605213).

## GD_lr1e-4_tiles30_seed42 / _seed1 (2026-10-15, jobs 1605114/15)
- granite lr 1e-4, 10 epochs, 30 tiles (longest_edge 3072). Val-chosen 0.536 / 0.614 and 0.596 / 0.620; 2-seed mean
  0.566 / 0.617 vs 12 tiles 0.616 / 0.616. Not adopted; weights deleted.

## SV_lr6e-5_tiles48 (2026-10-15, job 1605113)
- SmolVLM lr 6e-5, 10 epochs, 48 tiles (longest_edge 4096). Val-chosen ep9 0.638 / 0.647; epoch mean 0.637 / 0.640.
  Same as 30 tiles within spread; not adopted; weights deleted.

## GD_lr1e-4_tiles48_seed42 (2026-10-15, job 1605116)
- granite lr 1e-4, 10 epochs, 48 tiles. Val-chosen ep6 0.594 / 0.601; epoch mean 0.583 / 0.591. Not adopted; weights deleted.

## GD_soup4_lr1e-4 (2026-10-15, job 1605209)
- Weight average of 4 granite lr 1e-4 seeds (42 model, 1 ckpt-792, 2 ckpt-704, 3 ckpt-792), 12 tiles: 0.660 / 0.649. Granite best.

## SV_soup2_tiles30 (2026-10-15, job 1605213)
- Weight average of SV_lr6e-5_tiles30 (seed 42, ep10) and _seed1 (ep6), 30 tiles: 0.644 / 0.660. SmolVLM best on val.

## GD_lr1e-4_ep10_seed4 / _seed5 (2026-10-15, jobs 1605223/24)
- granite lr 1e-4, 10 epochs, 12 tiles, seeds 4 and 5 for a larger weight average. Val-chosen: seed 4 ep10 0.654 / 0.608;
  seed 5 ep7 0.567 / 0.610. Both kept as members of GD_soup6_lr1e-4.

## GD_soup6_lr1e-4 (2026-10-15, job 1605347)
- Weight average of 6 granite lr 1e-4 seeds: 0.675 / 0.629. Below the 4-seed average on val (0.649); not adopted.

## SV_lr6e-5_tiles30_seed2 (2026-10-15, job 1605225)
- SmolVLM lr 6e-5, 10 epochs, 30 tiles, seed 2. Val-chosen ep8 0.625 / 0.656; epoch mean 0.629 / 0.642.
  3-seed mean at 30 tiles (val-chosen): 0.619 / 0.654. Member of SV_soup3_tiles30.

## SV_soup3_tiles30 (2026-10-15, job 1605370)
- Weight average of 3 SmolVLM 30-tile seeds: 0.644 / 0.676. SmolVLM best.

## KW_ivl_k8 (2026-10-15, job 1605219)
- InternVL3-1B, key-by-key (--keywise 8), 3 epochs, lr 1e-5, 12 tiles, from the base model. Val-chosen ep3 0.647 / 0.650
  lenient (content 0.809 / 0.808); ep2 0.590 / 0.596. Below whole-page (0.705 / 0.708); weights deleted.

## KW_ivl_soupinit (2026-10-15, job 1605382)
- InternVL3-1B, key-by-key (--keywise 8), 1 epoch, starting from S_soup10_vision weights. 0.681 / 0.667 lenient
  (content 0.818 / 0.795) vs 0.728 / 0.732 for the start weights. Weights deleted.

## GD_kw_k8 (2026-10-15, job 1605221)
- granite-docling-258M, key-by-key (--keywise 8), 3 epochs, lr 1e-4, 12 tiles. Val-chosen ep3 0.474 / 0.463 lenient
  (content 0.640 / 0.615); ep2 0.316 / 0.342. Below whole-page (0.616 / 0.616); weights deleted.

## SV_kw_k8 (2026-10-15, job 1605220)
- SmolVLM-500M, key-by-key (--keywise 8), 3 epochs, lr 6e-5, 30 tiles. Val-chosen ep3 0.530 / 0.526 lenient
  (content 0.715 / 0.699); ep2 0.446 / 0.470. Below whole-page (0.619 / 0.654); weights deleted.

## KW_ivl_soupinit_ep3 (2026-10-15, job 1605482)
- InternVL3-1B, key-by-key (--keywise 8), 3 epochs from S_soup10_vision. Val-chosen ep3 0.711 / 0.704 lenient (content
  0.842 / 0.840); ep1 0.524 / 0.524, ep2 0.666 / 0.653. Start weights 0.728 / 0.732. Weights deleted.

## GD_neft15 (2026-10-15, job 1605666)
- granite-docling-258M, lr 1e-4, 10 epochs, 12 tiles, seed 42, NEFTune alpha 15 (one change vs GD_lr1e-4_ep10).
  Lenient gold / val: ep6 0.597 / 0.558, ep7 0.602 / 0.599, ep8 0.615 / 0.595, ep9 0.632 / 0.623, ep10 0.634 / 0.606.
  Val-chosen ep9 0.632 / 0.623 vs 0.609 / 0.596 same seed without noise. Seed-1 repeat queued (GD_neft15_seed1).

## SV_neft15_tiles30 (2026-10-15, job 1605665)
- SmolVLM-500M, lr 6e-5, 10 epochs, 30 tiles, seed 42, NEFTune alpha 15 (one change vs SV_lr6e-5_tiles30).
  Lenient gold / val: ep6 0.600 / 0.644, ep7 0.629 / 0.624, ep8 0.641 / 0.630, ep9 0.631 / 0.646, ep10 0.642 / 0.651.
  Val-chosen ep10 0.642 / 0.651 vs 0.645 / 0.656 without noise. No gain; weights deleted.

## KW_ivl_k8_ep6 (2026-10-15, job 1605424)
- InternVL3-1B, key-by-key (--keywise 8), 6 epochs, lr 1e-5, 12 tiles, from the base model. Val-chosen ep6 0.707 / 0.685
  lenient (content 0.846 / 0.821); ep5 0.679 / 0.673. Whole-page 9-seed mean 0.705 / 0.708. Weights deleted.

## GD_neft15_seed1 (2026-10-15, job 1605833)
- granite, NEFTune alpha 15, seed 1. Val-chosen ep9 0.613 / 0.610 vs 0.622 / 0.635 same seed without noise. Weights deleted.

## GD_neft15_seed2 (2026-10-15, job 1605861)
- granite, NEFTune alpha 15, seed 2. Val-chosen ep7 0.616 / 0.620 vs 0.621 / 0.601 same seed without noise.
  3-seed paired mean +0.003 / +0.007: NEFTune rejected. Weights of all granite noise runs deleted.

## SV_hk1k_tiles30 (2026-10-15, job 1605899)
- SmolVLM-500M, 30 tiles, lr 6e-5, 10 epochs, seed 42, + 993 hadikp pages (2,386 train pages). 2 h 41 min.
  Lenient gold / val: ep6 0.636 / 0.633, ep7 0.629 / 0.633, ep8 0.651 / 0.645, ep9 0.626 / 0.631, ep10 0.624 / 0.631.
  Val-chosen ep8 0.651 / 0.645 vs 0.645 / 0.656 without the pages. Rejected; weights deleted.
- Submitted: SV_lr6e-5_tiles30_seed3 / _seed4 (1605969/70) for a 5-seed average.

## GD_vlr1e-5 / _seed1 / _seed2 (2026-10-15, jobs 1605971/72/73)
- granite-docling-258M, lr 1e-4, vision encoder lr 1e-5 (`--vision-lr`, new), 10 epochs, 12 tiles. Seed 42 now; seeds 1/2 at 20:00.

## KW_ivl_soupinit_ep6 (2026-10-15, job 1605813)
- InternVL3-1B, key-by-key (--keywise 8), 6 epochs from S_soup10_vision. Val-chosen ep6 0.728 / 0.710 lenient (content
  0.854 / 0.825); ep5 0.716 / 0.692. Start weights 0.728 / 0.732. Key-by-key not adopted; weights deleted.

## ZV_vlr1e-6_seed42 / _seed1 / _seed2 (2026-10-15, queued for 20:00)
- InternVL3-1B default recipe with vision encoder lr 1e-6 (`--vision-lr`), main lr 1e-5, 6 epochs. Paired with Z/Z1/Z2_aug_vision.

## Disk cleanup (2026-10-15 ~15:20)
- Deleted weights of superseded or rejected runs (eval folders and logs kept): GD_soup2/soup6_lr1e-4, GD_lr1e-4_ep10_seed4/5,
  KW_smoke, SV_lr6e-5_ep10, SV_soup2_tiles30, ZH_hk241 (+seed1/2), ZK_hk1k (6 seeds), ZX_aug_vision_lr2e5, Z2 checkpoint-352.
  Group quota 1,920 → 1,867 GiB (soft limit 2,000). Kept: track bests, their members, SV 30-tile seeds for the 5-seed average.

## GD_vlr1e-5 (2026-10-15, job 1605971)
- granite, vision lr 1e-5 (main 1e-4), seed 42. Val-chosen ep8 0.547 / 0.577 vs 0.609 / 0.596. Weights deleted.
  GD_vlr1e-5_seed2 cancelled; ZV_vlr1e-6_seed42 resubmitted to start now (1606259).

## SV_lr6e-5_tiles30_seed4 (2026-10-15, job 1605970)
- SmolVLM-500M, 30 tiles, lr 6e-5, 10 epochs, seed 4. 2 h 30 min. Val-chosen ep8 0.641 / 0.659 lenient
  (ep9 0.657 / 0.641, ep10 0.647 / 0.645). Kept as a member of the planned 5-seed average.

## SV_lr6e-5_tiles30_seed3 (2026-10-15, job 1605969)
- SmolVLM-500M, 30 tiles, lr 6e-5, 10 epochs, seed 3. Val-chosen ep9 0.644 / 0.655 lenient. 5-seed mean (42/1/2/3/4)
  at val-chosen epochs: 0.627 / 0.656. Submitted SV_soup5_tiles30 (average of the 5 seeds at their val-chosen epochs).

## SV_soup5_tiles30 (2026-10-15, job 1606302)
- Weight average of 5 SmolVLM 30-tile seeds: 0.657 / 0.673 vs SV_soup3_tiles30 0.644 / 0.676. Not adopted.

## ZV_vlr1e-6_seed42 (2026-10-15, job 1606259)
- InternVL3-1B default recipe, vision lr 1e-6. Val-chosen ep5 0.725 / 0.682 vs 0.699 / 0.720. Not continued; seeds 1/2 cancelled; weights deleted.

## GD_vlr1e-5_seed1 (2026-10-15, job 1606140)
- granite, vision lr 1e-5, seed 1. Val-chosen ep8 0.597 / 0.597 vs 0.622 / 0.635. Rejected; weights deleted.

## Test set, milestone 1 (2026-10-15, jobs 1606330/31/32)
- Lenient F1 on 204 test pages: S_soup10_vision 0.745, SV_soup3_tiles30 0.677, GD_soup4_lr1e-4 0.633 (repaired 0.746 / 0.683 / 0.647).
- 2026-10-15 18:20 — deleted weights of SV_soup5_tiles30 (rejected) and its only-member seeds SV_lr6e-5_tiles30_seed3 / _seed4 (6 GB); eval folders kept. Group disk 1,890 of 2,000 GiB, mostly other projects.

### SV_tiles20_seed42 / _seed1 / _seed2 (TRUBA 1606349–51, ~2 h 15 min each) — SmolVLM, configs/smolvlm-500m.toml with --max-patches 20
- Val-chosen: seed 42 ep8 0.613 / 0.658; seed 1 ep8 0.617 / 0.648; seed 2 ep9 0.602 / 0.655 → **mean 0.611 / 0.654**.
- 30-tile reference, 3-seed mean 0.619 / 0.654: equal within spread. Next: SV_soup3_tiles20 (1606478, average of the
  three val-chosen checkpoints) vs SV_soup3_tiles30 0.644 / 0.676, then CPU time per page if they tie.
- SV_soup3_tiles20 (1606478, 8 min): **0.660 / 0.676** vs SV_soup3_tiles30 0.644 / 0.676 — tie (gold +0.016 within
  spread, val equal). Decision by CPU time on the same 10 val pages (both averages, M2 Max, float32, batch 1).
  Note: the README's SmolVLM CPU figure (59 s/page) was measured at 12 tiles, not the released 30; this timing replaces it.
- CPU timing, SmolVLM averages, 10 val pages, M2 Max, float32, batch 1: 20 tiles median 66.5 s (35–100), total 643 s,
  18.7 tokens/s, 4.7 GB, 8/10 valid; 30 tiles median 67.5 s (52–89), total 669 s, 17.1 tokens/s, 5.3 GB, 8/10 valid.
  20 tiles is 9 % faster per token but writes 5 % more tokens: 4 % less time overall. **Decision: keep the 30-tile
  average** (dev tie, speed within 4 %, and it is the model scored on the test pages). README / model card SmolVLM
  CPU figure corrected 59 s → 68 s (59 s was a 12-tile model from 2026-10-14).

### ZP_noperturb_s42 / _s1 / _s2 (TRUBA 1606401, 1606540–41, ~65 min each) — InternVL default recipe with --perturb 0
- Val-chosen: s42 ep6 0.718 / 0.711; s1 ep6 0.721 / 0.706; s2 ep6 0.708 / 0.715 → **mean 0.716 / 0.711**.
- Paired default seeds (perturb 0.1): 0.699 / 0.720, 0.722 / 0.711, 0.697 / 0.693 → mean 0.706 / 0.708.
- Epoch 6, gold + val, all 3 seeds: unparsed 15 vs 19, token-limit hits 8 vs 8. **No measurable effect**; the
  lora_v1 → v2 loop fix came from removing `corrections`. Perturb 0.1 stays in the recipe (no reason to change
  trained models); stronger perturbation (0.9) is tested separately on granite / SmolVLM. Weights deleted.

## GD_perturb0.9_s42 / _s1 / _s2 (2026-10-15, jobs 1606605/06/1606651, ~2 h 05 min each)
- granite-docling recipe with `--perturb 0.9` (~9 swapped input tokens per page). Val-chosen: s42 ep9 0.544 / 0.591,
  s1 ep9 0.615 / 0.605, s2 ep8 0.591 / 0.629; mean 0.583 / 0.608 vs lr1e-4 seeds 0.617 / 0.611. Not adopted; weights deleted.

## G4_grpo_unseen / G5_grpo_unseen_content (2026-10-15, jobs 1606603/04, ~2 h 49 min each)
- GRPO from Z_aug_vision checkpoint-528 (0.705 / 0.719), lr 3e-6, 4 samples, 974 unseen hadikp pages. Lenient reward
  0.708 / 0.714; content reward 0.718 / 0.712. Not adopted; weights deleted.

## GD_noperturb_s42 / _s1 / _s2 (2026-10-15, jobs 1606954/55/84, ~2 h each)
- granite-docling recipe with `--perturb 0`. Val-chosen: s42 ep10 0.618 / 0.629, s1 ep7 0.634 / 0.620, s2 ep7 0.577 / 0.606;
  mean 0.610 / 0.618 vs 0.617 / 0.611. No effect; weights deleted.

## SV_perturb0.9_s42 / _s1 / _s2 and SV_soup3_perturb0.9 (2026-10-15, jobs 1606607/52, 1606751, 1607307)
- SmolVLM 30-tile recipe with `--perturb 0.9`. Val-chosen: s42 ep8 0.616 / 0.665, s1 ep8 0.631 / 0.663, s2 ep7 0.636 / 0.668;
  mean 0.628 / 0.665 vs 0.619 / 0.654. Average of the three: 0.674 / 0.658 vs SV_soup3_tiles30 0.644 / 0.676. Not adopted.

## S_soup4_clean (2026-10-16, job 1608049; test 1608124)
- Average of Z_aug_vision ckpt-528, Z1_aug_vision_seed1, Z2_aug_vision_seed2, Z3_aug_vision_seed3 ckpt-440 (no hadikp
  pages). Dev 0.734 / 0.712. Test lenient 0.739 (0.746 repaired), strict 0.726, content 0.841 vs S_soup10_vision 0.745 (0.746).

## ZR_seed4 … ZR_seed9 (2026-10-16, jobs 1608125–30, ~1 h each)
- Default recipe configs/internvl3-1b.toml, seeds 4–9, labels_train_v6 only: members for the clean 10-model average.
  Val-chosen: seed4 ep5 0.724 / 0.713, seed5 ep5 0.698 / 0.700, seed6 ep5 0.701 / 0.715, seed7 ep6 0.721 / 0.711, seed8 ep5 0.695 / 0.696,
  seed9 ep5 0.726 / 0.709. Mean 0.711 / 0.707.

## S_soup10_clean (2026-10-16, job 1608784; test 1608785) — released as ResuVL-L
- Average of the 4 S_soup4_clean members + ZR_seed4–9 at val-chosen epochs. Dev 0.732 / 0.724 lenient, content
  0.847 / 0.855. Test lenient 0.735 (0.739 repaired), strict 0.722, content 0.840; 4 of 204 unparsed.

## N35_seed42 / _seed1 / _seed2 (2026-10-16, jobs 1608960–62, ~1 h 20 min each)
- InternVL3.5-1B-HF (1.06B) with configs/internvl3-1b.toml unchanged. Val-chosen: seed42 ep6 0.721 / 0.703, seed1 ep4
  0.723 / 0.707, seed2 ep5 0.691 / 0.715; mean 0.712 / 0.708 vs InternVL3-1B singles 0.711 / 0.707. No gain.
