"""Command-line entry point: tiny-resume-parser COMMAND [options]; COMMAND --help shows each command's options."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "parse": ("parse", "read CV images or PDFs with a trained model and write JSON"),
    "train": ("train", "fine-tune a student on teacher-labelled pages"),
    "evaluate": ("evaluate", "generate answers for labelled pages and score them"),
    "rescore": ("rescore", "score saved answers with strict / lenient / content F1"),
    "soup": ("soup", "average the weights of several fine-tuned models"),
    "label": ("label", "label CV pages with the teacher model (GLM-5.3-Flash)"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: tiny-resume-parser COMMAND [options]\n\ncommands:")
        for name, (_, text) in COMMANDS.items():
            print(f"  {name:<9} {text}")
        raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    module = importlib.import_module(f"tiny_resume_parser.{COMMANDS[sys.argv[1]][0]}")
    sys.argv = [f"tiny-resume-parser {sys.argv[1]}", *sys.argv[2:]]
    result = module.main(sys.argv[1:])
    raise SystemExit(result if isinstance(result, int) else 0)
