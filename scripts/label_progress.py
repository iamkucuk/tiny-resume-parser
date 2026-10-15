"""Progress of a running tiny_resume_parser.label job, read from its output JSONL, log and PID file.

    python scripts/label_progress.py data/processed/labels_full_v6.jsonl --ids data/processed/cv_ids_full_size.json

Prints: whether the process is alive, pages done / total, ok and failed counts, speed over the
last hour, estimated time left, retries and rate-limit waits, and the latest failures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"


def _alive(pid_path: Path) -> str:
    if not pid_path.exists():
        return "unknown (no PID file)"
    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return f"NOT RUNNING (PID {pid} has exited)"
    return f"running (PID {pid})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show progress of an tiny_resume_parser.label run.")
    ap.add_argument("output", type=Path, help="the JSONL the run writes to")
    ap.add_argument(
        "--ids", type=Path, help="JSON list of cv_ids the run was limited to (tiny_resume_parser.label --ids)"
    )
    args = ap.parse_args()

    docs = [json.loads(line) for line in DOCUMENTS_PATH.read_text().splitlines()]
    ids = set(json.loads(args.ids.read_text())) if args.ids else {d["cv_id"] for d in docs}
    total = sum(len(d["pages"]) for d in docs if d["cv_id"] in ids)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    rows = [r for r in rows if r["cv_id"] in ids]
    latest = {(r["cv_id"], r["page"]): r for r in rows}  # reruns append; last line per page wins
    status = Counter(r["status"] for r in latest.values())
    n_ok = status["ok"]

    now = time.time()
    recent = [r for r in rows if now - r["ts"] < 3600]
    span_h = (now - min(r["ts"] for r in recent)) / 3600 if len(recent) > 1 else 0
    per_hour = len(recent) / span_h if span_h else 0
    left = total - n_ok

    attempts = [a for r in rows for a in r.get("attempts", [])]
    att = Counter(a["status"] for a in attempts)
    retried = sum(1 for r in latest.values() if sum(a["status"] != "rate_limited" for a in r.get("attempts", [])) > 1)

    print(f"process:        {_alive(args.output.with_suffix('.pid'))}")
    print(f"pages ok:       {n_ok} / {total} ({100 * n_ok / total:.1f}%)")
    print(f"pages not ok:   {sum(v for k, v in status.items() if k != 'ok')} {dict(status - Counter(ok=n_ok)) or ''}")
    print(f"speed:          {per_hour:.0f} pages/hour over the last hour")
    print(
        f"time left:      {left / per_hour:.1f} h"
        if per_hour
        else "time left:      unknown (no pages finished in the last hour)"
    )
    print(f"pages retried:  {retried}; attempt outcomes: {dict(att)}")
    if rows:
        print(
            f"last page:      {time.strftime('%H:%M:%S', time.localtime(rows[-1]['ts']))} {rows[-1]['cv_id']} p{rows[-1]['page']}"
        )
    for r in [r for r in latest.values() if r["status"] != "ok"][-5:]:
        last = (r.get("attempts") or [{}])[-1]
        print(
            f"  failed: {r['cv_id']} p{r['page']} {r['status']}: {last.get('status')} {str(last.get('errors'))[:150]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
