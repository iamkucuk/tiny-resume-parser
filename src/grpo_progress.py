"""Reward trend of a running GRPO job, from the per-step dicts TRL prints to the job's .out log.

    python src/grpo_progress.py /arf/scratch/fkucuk/cv_parser_test/logs/G1_grpo_lr1e-6-1602542.out [--block 25]
"""
import argparse
import ast

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--block", type=int, default=25)
args = ap.parse_args()
rows = [ast.literal_eval(line.strip()) for line in open(args.log) if line.startswith("{'loss'") and "'reward'" in line]
for i in range(0, len(rows), args.block):
    b = rows[i:i + args.block]
    m = {k: sum(r[k] for r in b) / len(b) for k in ("reward", "reward_std", "completions/mean_length",
                                                    "completions/clipped_ratio", "frac_reward_zero_std")}
    print(f"steps {i + 1}-{i + len(b)}: reward {m['reward']:.3f}  std in group {m['reward_std']:.3f}  "
          f"length {m['completions/mean_length']:.0f}  cut at limit {m['completions/clipped_ratio']:.3f}  "
          f"groups with equal rewards {m['frac_reward_zero_std']:.2f}")
