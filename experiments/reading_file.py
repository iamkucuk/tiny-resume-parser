"""Build the reading-stage training file from granite-docling transcripts (experiments/transcribe.py shards).

    python experiments/reading_file.py --out data/processed/reading_train.jsonl data/processed/transcripts_s*.jsonl

Cleaning: a transcript is cut before the third copy of any line (granite-docling loops repeat lines until the token
limit; a third copy is where the cut costs little real text), then dropped if under 300 characters (empty pages:
granite-docling returned the page as one picture) or if under 80 % of its lines are unique. Prints the counts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("shards", type=Path, nargs="+")
args = ap.parse_args()

rows = [json.loads(line) for p in args.shards for line in p.read_text().splitlines()]
counts, kept = Counter(), []
for r in rows:
    seen, lines = Counter(), []
    for line in r["transcript"].splitlines():
        seen[line] += 1
        if seen[line] >= 3:
            break
        lines.append(line)
    text = "\n".join(lines)
    if len(text) < 300:
        counts["dropped: under 300 characters"] += 1
    elif len(set(lines)) / len(lines) < 0.8:
        counts["dropped: under 80 % unique lines"] += 1
    else:
        counts["kept, cut at a third copy" if len(lines) < len(r["transcript"].splitlines()) else "kept whole"] += 1
        kept.append({k: r[k] for k in ("cv_id", "page", "num_pages", "image")} | {"transcript": text})
args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
print(f"{len(rows)} transcripts -> {len(kept)} kept in {args.out}: {dict(counts)}")
