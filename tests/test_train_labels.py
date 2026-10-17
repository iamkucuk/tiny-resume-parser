"""What the model learns from (tiny_resume_parser/train.py Collator): only the answer and its end token are learnt,
never the prompt or the image tokens, and token perturbation changes the input but never the labels. A wrong mask
trains silently and only shows up as lower scores, so it is checked here with a small character-level stand-in for
the Qwen3.5 processor instead of a downloaded model."""

import random
import re

import torch

import tiny_resume_parser.train as train
from tiny_resume_parser.models import student_target

SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001, "<image>": 1002}


class Tokenizer:
    vocab_size = 1000
    added_tokens_decoder = {i: t for t, i in SPECIAL.items()}

    def encode(self, text: str) -> list[int]:
        parts = re.split("(" + "|".join(map(re.escape, SPECIAL)) + ")", text)
        return [i for p in parts for i in ([SPECIAL[p]] if p in SPECIAL else [ord(c) for c in p])]

    def __call__(self, text, add_special_tokens=False):
        return type("Encoding", (), {"input_ids": self.encode(text)})

    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIAL[token]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.added_tokens_decoder.get(i) or chr(i) for i in ids)


class Qwen3VLProcessor:  # the class name selects the Qwen3.5 header and end token in models.FAMILY
    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, tokenize=False):
        user, answer = messages[0]["content"][1]["text"], messages[1]["content"][0]["text"]
        return (
            f"<|im_start|>user\n<image><image>{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n"
        )

    def __call__(self, text, images, padding, return_tensors, **size):
        ids = [self.tokenizer.encode(t) for t in text]
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([i + [0] * (width - len(i)) for i in ids])}


RECORDS = [{"basics": {"name": "Jean Picard"}}, {"skills": [{"name": "SQL"}, {"name": "Python"}]}]


def collate(monkeypatch, perturb: float = 0.0) -> dict:
    monkeypatch.setattr(train, "page_image", lambda row: None)
    rows = [{"record": r, "cv_id": "cv", "page": 1} for r in RECORDS]
    return train.Collator(Qwen3VLProcessor(), perturb=perturb)(rows)


def test_only_the_answer_and_its_end_token_are_learnt(monkeypatch):
    batch = collate(monkeypatch)
    tok = Tokenizer()
    for record, labels in zip(RECORDS, batch["labels"].tolist()):
        learnt = [i for i in labels if i != -100]
        assert tok.decode(learnt) == student_target(record) + "<|im_end|>"


def test_perturbation_changes_the_input_but_not_the_labels(monkeypatch):
    random.seed(0)  # perturbation draws positions and tokens at random
    clean, noisy = collate(monkeypatch), collate(monkeypatch, perturb=0.9)
    assert torch.equal(clean["labels"], noisy["labels"])
    changed = clean["input_ids"] != noisy["input_ids"]
    assert changed.any() and (clean["labels"][changed] != -100).all()  # only answer tokens are replaced
    assert not set(noisy["input_ids"][changed].tolist()) & set(SPECIAL.values())
