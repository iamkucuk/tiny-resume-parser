"""Average the weights of several full fine-tunes of the same base model (a uniform "model soup", arXiv 2203.05482).

    python -m tiny_resume_parser.soup --out runs/S_soup4 runs/Z_aug_vision/checkpoint-528 runs/Z1_aug_vision_seed1/model ...

Each input folder holds config.json + model.safetensors (a Trainer checkpoint or a saved model/). The output folder
gets the averaged model.safetensors and the first input's config files, so evaluate.py --model <out> loads it.
"""

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("inputs", type=Path, nargs="+")
    args = ap.parse_args(argv)

    total, dtypes = None, {}
    for p in args.inputs:
        sd = load_file(p / "model.safetensors")
        if total is None:
            total = {k: v.float() for k, v in sd.items()}
            dtypes = {k: v.dtype for k, v in sd.items()}
        else:
            if sd.keys() != total.keys():
                raise SystemExit(f"{p}: tensor names differ from {args.inputs[0]}")
            for k, v in sd.items():
                total[k] += v.float()
        print(f"added {p} ({len(sd)} tensors)")
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: (v / len(args.inputs)).to(dtypes[k]).contiguous() for k, v in total.items()},
        str(args.out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    for name in ("config.json", "generation_config.json"):
        shutil.copy(args.inputs[0] / name, args.out / name)
    print(f"saved average of {len(args.inputs)} models to {args.out}")


if __name__ == "__main__":
    main()
