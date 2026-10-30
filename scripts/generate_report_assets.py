"""Build reproducible report tables, plots, and qualitative examples from run logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "outputs" / "runs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RUNS = {
    "Scratch Transformer": "transformer_full_20261029",
    "mT5-small": "mt5_small_20261029_r3",
    "LLaMA 3.2 1B LoRA": "llama_lora_20261030_r2",
    "Gemma 3 270M LoRA": "gemma_lora_20261030",
}

QUALITATIVE_IDS = [0, 147, 179, 500, 2500]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    summaries = {name: read_json(RUNS_DIR / run / "summary.json") for name, run in RUNS.items()}
    configs = {name: read_json(RUNS_DIR / run / "run_config.json") for name, run in RUNS.items()}

    comparison_rows = []
    for name, run in RUNS.items():
        summary = summaries[name]
        config = configs[name]
        comparison_rows.append(
            {
                "model": name,
                "run_id": run,
                "adaptation": summary.get("adaptation", "Full training" if name == "Scratch Transformer" else "Full fine-tuning"),
                "epochs_completed": summary["epochs_completed"],
                "best_epoch": summary["best_epoch"],
                "total_parameters": summary["total_parameters"],
                "trainable_parameters": summary["trainable_parameters"],
                "trainable_percent": 100 * summary["trainable_parameters"] / summary["total_parameters"],
                "training_seconds": summary["training_seconds"],
                "training_hours": summary["training_seconds"] / 3600,
                "inference_seconds": summary["inference_seconds"],
                "seconds_per_test_sentence": summary["seconds_per_test_sentence"],
                "test_bleu": summary["test_bleu"],
                "test_chrf": summary["test_chrf"],
                "test_sentences": summary["test_sentences"],
                "peak_gpu_memory_gib": max(item["gpu_peak_allocated_bytes"] for item in summary["history"]) / (1024**3),
                "decoding": config["decoding"]["method"],
            }
        )

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    epoch_rows = []
    for name, run in RUNS.items():
        for item in summaries[name]["history"]:
            epoch_rows.append(
                {
                    "model": name,
                    "run_id": run,
                    "epoch": item["epoch"],
                    "train_loss": item["train_loss"],
                    "validation_loss": item["validation_loss"],
                    "validation_bleu": item["validation_bleu"],
                    "validation_chrf": item["validation_chrf"],
                    "epoch_seconds": item["epoch_seconds"],
                    "gpu_peak_allocated_bytes": item["gpu_peak_allocated_bytes"],
                }
            )
    with (RESULTS_DIR / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=epoch_rows[0].keys())
        writer.writeheader()
        writer.writerows(epoch_rows)

    names = list(RUNS)
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].bar(names, [summaries[n]["test_bleu"] for n in names], color=colors)
    axes[0].set_title("Corpus BLEU")
    axes[0].set_ylabel("BLEU score")
    axes[1].bar(names, [summaries[n]["test_chrf"] for n in names], color=colors)
    axes[1].set_title("Corpus chrF")
    axes[1].set_ylabel("chrF score")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("English to Hindi test-set quality", fontsize=14, fontweight="bold")
    save_figure(FIGURES_DIR / "quality_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].bar(names, [summaries[n]["training_seconds"] / 3600 for n in names], color=colors)
    axes[0].set_title("Training time")
    axes[0].set_ylabel("Hours")
    axes[1].bar(names, [summaries[n]["seconds_per_test_sentence"] for n in names], color=colors)
    axes[1].set_title("Greedy decoding latency")
    axes[1].set_ylabel("Seconds per sentence")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Measured computational cost", fontsize=14, fontweight="bold")
    save_figure(FIGURES_DIR / "runtime_comparison.png")

    x = range(len(names))
    total = [summaries[n]["total_parameters"] for n in names]
    trainable = [summaries[n]["trainable_parameters"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    width = 0.36
    ax.bar([i - width / 2 for i in x], total, width, label="Total", color="#4472C4")
    ax.bar([i + width / 2 for i in x], trainable, width, label="Trainable", color="#ED7D31")
    ax.set_yscale("log")
    ax.set_ylabel("Parameters log scale")
    ax.set_xticks(list(x), names, rotation=20)
    ax.set_title("Total and trainable parameter counts", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}M"))
    save_figure(FIGURES_DIR / "parameter_comparison.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for name in names:
        history = summaries[name]["history"]
        ax.plot([x["epoch"] for x in history], [x["validation_loss"] for x in history], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss by epoch", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(FIGURES_DIR / "validation_loss.png")

    predictions = {
        name: {row["id"]: row for row in read_jsonl(RUNS_DIR / run / "test_predictions.jsonl")}
        for name, run in RUNS.items()
    }
    qualitative = []
    for sample_id in QUALITATIVE_IDS:
        anchor = predictions[names[0]][sample_id]
        qualitative.append(
            {
                "id": sample_id,
                "source": anchor["source"],
                "reference": anchor["reference"],
                "predictions": {name: predictions[name][sample_id]["prediction"] for name in names},
            }
        )
    (RESULTS_DIR / "qualitative_samples.json").write_text(
        json.dumps(qualitative, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Qualitative Translation Samples",
        "",
        "These fixed test examples show both the strongest observed behavior and recurring failure modes.",
        "",
    ]
    for sample in qualitative:
        lines.extend(
            [
                f"## Test item {sample['id']}",
                "",
                f"**English:** {sample['source']}",
                "",
                f"**Reference:** {sample['reference']}",
                "",
            ]
        )
        for name, prediction in sample["predictions"].items():
            lines.extend([f"**{name}:** {prediction or '[empty output]'}", ""])
    (RESULTS_DIR / "qualitative_samples.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote report assets to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
