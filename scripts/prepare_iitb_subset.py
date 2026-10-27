"""Create the fixed IIT Bombay English-Hindi subset used by this project.

The IITB corpus is treated as an already prepared parallel corpus. This script
only checks that paired files have the same number of lines, selects a seeded
random sample from the official training split, and copies the official dev
and test splits unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from itertools import zip_longest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible 30,000-pair IITB training subset."
    )
    parser.add_argument(
        "--train-en",
        type=Path,
        default=PROJECT_ROOT / "parallel" / "parallel-n" / "IITB.en-hi.en",
    )
    parser.add_argument(
        "--train-hi",
        type=Path,
        default=PROJECT_ROOT / "parallel" / "parallel-n" / "IITB.en-hi.hi",
    )
    parser.add_argument(
        "--dev-en",
        type=Path,
        default=PROJECT_ROOT / "dev_test" / "dev_test" / "dev.en",
    )
    parser.add_argument(
        "--dev-hi",
        type=Path,
        default=PROJECT_ROOT / "dev_test" / "dev_test" / "dev.hi",
    )
    parser.add_argument(
        "--test-en",
        type=Path,
        default=PROJECT_ROOT / "dev_test" / "dev_test" / "test.en",
    )
    parser.add_argument(
        "--test-hi",
        type=Path,
        default=PROJECT_ROOT / "dev_test" / "dev_test" / "test.hi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "iitb_en_hi_30k",
    )
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def sample_parallel(
    english_path: Path,
    hindi_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[tuple[int, str, str]], int]:
    """Reservoir-sample aligned records in one streaming pass."""
    if sample_size <= 0:
        raise ValueError("sample size must be positive")

    rng = random.Random(seed)
    reservoir: list[tuple[int, str, str]] = []
    missing = object()
    total = 0

    with english_path.open("r", encoding="utf-8") as english_file, hindi_path.open(
        "r", encoding="utf-8"
    ) as hindi_file:
        for index, pair in enumerate(
            zip_longest(english_file, hindi_file, fillvalue=missing)
        ):
            english, hindi = pair
            if english is missing or hindi is missing:
                raise ValueError(
                    "Training files are not aligned: their line counts differ."
                )

            record = (
                index,
                english.rstrip("\r\n"),
                hindi.rstrip("\r\n"),
            )
            if index < sample_size:
                reservoir.append(record)
            else:
                replacement = rng.randint(0, index)
                if replacement < sample_size:
                    reservoir[replacement] = record
            total = index + 1

    if total < sample_size:
        raise ValueError(
            f"Requested {sample_size:,} pairs, but the corpus has only {total:,}."
        )

    rng.shuffle(reservoir)
    return reservoir, total


def count_aligned(english_path: Path, hindi_path: Path) -> int:
    missing = object()
    count = 0
    with english_path.open("r", encoding="utf-8") as english_file, hindi_path.open(
        "r", encoding="utf-8"
    ) as hindi_file:
        for pair in zip_longest(english_file, hindi_file, fillvalue=missing):
            if pair[0] is missing or pair[1] is missing:
                raise ValueError(
                    f"Files are not aligned: {english_path} and {hindi_path}"
                )
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_training_subset(
    records: list[tuple[int, str, str]], output_dir: Path
) -> dict[str, Path]:
    output_paths = {
        "train_en": output_dir / "train.en",
        "train_hi": output_dir / "train.hi",
        "train_indices": output_dir / "train_indices.txt",
    }
    with output_paths["train_en"].open("w", encoding="utf-8", newline="\n") as en_out, output_paths[
        "train_hi"
    ].open("w", encoding="utf-8", newline="\n") as hi_out, output_paths[
        "train_indices"
    ].open("w", encoding="utf-8", newline="\n") as index_out:
        for index, english, hindi in records:
            en_out.write(f"{english}\n")
            hi_out.write(f"{hindi}\n")
            index_out.write(f"{index}\n")
    return output_paths


def main() -> None:
    args = parse_args()
    input_paths = [
        args.train_en,
        args.train_hi,
        args.dev_en,
        args.dev_hi,
        args.test_en,
        args.test_hi,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, training_population = sample_parallel(
        args.train_en, args.train_hi, args.sample_size, args.seed
    )
    dev_count = count_aligned(args.dev_en, args.dev_hi)
    test_count = count_aligned(args.test_en, args.test_hi)

    outputs = write_training_subset(records, args.output_dir)
    copied_files = {
        "dev_en": args.output_dir / "dev.en",
        "dev_hi": args.output_dir / "dev.hi",
        "test_en": args.output_dir / "test.en",
        "test_hi": args.output_dir / "test.hi",
    }
    for source, destination in (
        (args.dev_en, copied_files["dev_en"]),
        (args.dev_hi, copied_files["dev_hi"]),
        (args.test_en, copied_files["test_en"]),
        (args.test_hi, copied_files["test_hi"]),
    ):
        shutil.copyfile(source, destination)
    outputs.update(copied_files)

    manifest = {
        "dataset": "IIT Bombay English-Hindi Parallel Corpus v3.1",
        "protocol": "seeded reservoir sample; no text preprocessing",
        "seed": args.seed,
        "training_population_pairs": training_population,
        "splits": {
            "train": len(records),
            "validation": dev_count,
            "test": test_count,
        },
        "files": {
            name: {
                "name": path.name,
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training population: {training_population:,} pairs")
    print(f"Created train split: {len(records):,} pairs")
    print(f"Copied validation split: {dev_count:,} pairs")
    print(f"Copied test split: {test_count:,} pairs")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
