# Findings

This page summarises what changed the score and what did not, in the order the work happened. Every change was
compared with a named baseline run, usually over three training seeds, and was kept only if it beat the
seed-to-seed spread (about 0.012 F1) on both development sets: 50 reviewed gold pages and 79 validation pages.
Scores below are lenient field F1 on those two sets, written as gold / validation. The full record of every run,
including the reasoning before each one, is in the [research log](research/README.md).

## How the recipe developed

**Getting a valid answer at all.** The first LoRA fine-tune of InternVL3-1B reached a low training loss, yet none
of 10 answers was valid JSON: every answer repeated one entry of the label's `corrections` list until the
2,500-token limit. That list recorded spelling fixes the labelling model had made, so its content was not visible
on the page. Removing it from the target, together with token perturbation (on about one page in ten, one input
token of the answer is replaced by a random token while the target stays correct; Nougat, arXiv 2308.13418),
raised valid answers to 21 of 50. The two changes were made in the same run, so their separate effects are unknown;
a test of token perturbation alone is running. The lesson that stayed: training loss did not show the loops; only
generating answers did.

**LoRA, then full fine-tuning.** More epochs, a trainable image-to-text projector and 12 image tiles instead of 6
brought LoRA to about 0.65 on the gold pages, where it stopped improving. The 12 tiles helped tall pages (+0.094 on
the 6 such gold pages) and nothing else. Fine-tuning all weights instead gave 0.693 / 0.669, about +0.03.

**Image noise and the vision encoder.** Training on pages with mild noise (lower resolution, rotation up to 1.5°,
contrast, blur, JPEG artefacts, on half of the pages) and also training the vision encoder raised the 3-seed mean
from 0.684 / 0.671 to 0.706 / 0.708. Each change alone helped only one of the two sets.

**Weight averaging.** Averaging the weights of 10 fine-tuned models ("model soup", arXiv 2203.05482) gave
0.728 / 0.732, +0.024 on both sets over the mean of single models, at no extra inference cost. Larger averages (13 to 19
models) did not score higher on validation, so the 10-model average is the released InternVL model.

**Answer repair.** On 3 of the 50 gold pages the best model writes no valid JSON (a loop, a bad escape, a missing
comma), and those pages hold 6.7 % of all gold values. Repairing the answer after generation (fix escapes, keep the
complete values before the break, drop repeated list items) raises gold from 0.728 to 0.746 and leaves validation
unchanged. Development decisions compare unrepaired answers; the `parse` command returns repaired ones.

**More data.** Training on half of the pages cost 0.04 to 0.08 with the same number of steps, so the number of
distinct pages matters. Adding 993 pages from a second dataset gave +0.011 / +0.013 over 6 seeds, which is within
noise; its licence is unknown, so released weights do not use it.

**The two smaller models.** The InternVL recipe did not transfer. SmolVLM-500M with it scored 0.370 / 0.460, and 10
of its 50 gold answers looped to the token limit. A 6× higher learning rate and 10 epochs gave 0.596 / 0.627; 30
image tiles instead of 12 added +0.043 / +0.032; averaging 3 seeds gave 0.644 / 0.676. granite-docling-258M needed
an even higher learning rate (1e-4: +0.024 / +0.031 over 6e-5), did worse with more than 12 tiles, and its 4-seed
average reached 0.660 / 0.649, +0.044 / +0.033 over a single model.

## What did not help

| Tried | Result |
|---|---|
| Preference training (DPO) on the model's own wrong answers | gold fell at every checkpoint, from 0.650 to 0.539 |
| Reinforcement learning (GRPO) with field F1 as the reward, 4 settings | 0.705–0.718 / 0.704–0.718, starting from 0.705 / 0.719 |
| Generating one JSON key per call (29 calls per page) | at best 0.728 / 0.710 against 0.728 / 0.732 |
| Decoding constrained to the schema | 0.694 against 0.693, 5.4× slower; useful only to guarantee valid JSON |
| Repetition penalty 1.02 or 1.05 | no gain on InternVL; lower on SmolVLM (0.342 / 0.430 against 0.370 / 0.460) |
| Banning repeated word sequences (n-gram ban) | broke normal JSON: 3 of 10 answers valid at best |
| Training on labels rewritten into a canonical form | −0.01 |
| 24 image tiles for InternVL | −0.05 on gold |
| 2 pages per step instead of 1; other learning rates; 8 epochs | no gain or lower |
| NEFTune noise on text embeddings (arXiv 2310.05914) | +0.003 / +0.007 on granite over 3 paired seeds; −0.003 / −0.005 on SmolVLM |
| A 10× lower learning rate for the vision encoder | lower on granite-docling (2 seeds); +0.026 / −0.038 on InternVL |
| A first training stage on page transcripts (SmolVLM) | +0.005 / −0.023 |
| A second model checking and fixing the labels (30 pages) | fixed 14 real errors, introduced about 8 new ones |

Reinforcement learning raised its training reward but not the development scores. Its training logs and answers
show why in more detail; that analysis is in the research log entry of 2026-10-15.

## Where the errors are

<img alt="Bar chart of gold values the best InternVL model loses: left out 10.6 %, page not parsed 6.7 %, joined with other text 4.6 %, written in part 3.1 %, misread characters 2.9 %, under another field 1.8 %." src="assets/lost-values.svg" width="700">

The largest loss is values the model does not write at all. Misread characters are one of the smallest. Part of the
remaining gap is in the labels themselves. When the labelling model labels 50 reviewed pages a second time, it
scores 0.912 field F1 against them; about half of the disagreements are the same text split into fields
differently. Measured by content F1, which ignores how text is split, the second run reaches 0.967.

Model size is the strongest lever measured, but it is outside this project's limit of 1B parameters: with LoRA,
InternVL3-2B reached 0.754 / 0.716 and InternVL3-8B 0.824 / 0.812.

## Open questions

- The separate effects of removing the `corrections` list and of token perturbation (a test of the second is running).
- Whether SmolVLM can use 20 image tiles instead of 30: one seed each scored the same, a 3-seed test is running.
- Why granite-docling loses accuracy above 12 tiles, and why 2 pages per step scored lower.
- Merging the pages of one CV, and faster CPU inference through quantisation.
