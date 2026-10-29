"""Shared data loading, metrics, and timing helpers for the MT experiments."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iitb_en_hi_30k"
OUT = ROOT / "outputs"
SEED = 1729


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def split_data(name: str) -> tuple[list[str], list[str]]:
    return read_lines(DATA / f"{name}.en"), read_lines(DATA / f"{name}.hi")


def evaluate(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "bleu": BLEU().corpus_score(predictions, [references]).score,
        "chrf": CHRF().corpus_score(predictions, [references]).score,
    }


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def timed_generate(generate, sources: list[str], batch_size: int = 8) -> tuple[list[str], float]:
    predictions: list[str] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(sources), batch_size):
        predictions.extend(generate(sources[offset : offset + batch_size]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - started


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
