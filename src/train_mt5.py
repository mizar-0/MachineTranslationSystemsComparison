"""Fine-tune mT5-small for English-to-Hindi translation with report-grade logs."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import OUT, SEED, evaluate, save_json, seed_everything, split_data, timed_generate


class TranslationDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_length, max_target_length):
        self.rows = []
        self.source_truncated = 0
        self.target_truncated = 0
        for source, target in zip(sources, targets):
            prompted = f"translate English to Hindi: {source}"
            full_source_ids = tokenizer(prompted, add_special_tokens=True, truncation=False)["input_ids"]
            full_target_ids = tokenizer(text_target=target, add_special_tokens=True, truncation=False)["input_ids"]
            self.source_truncated += len(full_source_ids) > max_source_length
            self.target_truncated += len(full_target_ids) > max_target_length
            source_ids = tokenizer(
                prompted, add_special_tokens=True, truncation=True,
                max_length=max_source_length,
            )["input_ids"]
            target_ids = tokenizer(
                text_target=target, add_special_tokens=True, truncation=True,
                max_length=max_target_length,
            )["input_ids"]
            self.rows.append((source_ids, target_ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(tokenizer):
    def collate(rows):
        sources, targets = zip(*rows)
        source_batch = tokenizer.pad(
            {"input_ids": list(sources)}, padding=True, return_tensors="pt",
        )
        max_target = max(len(ids) for ids in targets)
        labels = [ids + [-100] * (max_target - len(ids)) for ids in targets]
        source_batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return source_batch
    return collate


def generate(model, tokenizer, sources, max_source_length, max_new_tokens):
    prompts = [f"translate English to Hindi: {source}" for source in sources]
    batch = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_source_length,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **batch, do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    run_dir = OUT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    save_json(run_dir / "status.json", {"state": "downloading_model", "started_at_utc": started_at})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    train_en, train_hi = split_data("train")
    dev_en, dev_hi = split_data("dev")
    test_en, test_hi = split_data("test")
    train_en, train_hi = train_en[:args.max_train], train_hi[:args.max_train]
    train_dataset = TranslationDataset(
        train_en, train_hi, tokenizer, args.max_source_length, args.max_target_length,
    )
    dev_dataset = TranslationDataset(
        dev_en, dev_hi, tokenizer, args.max_source_length, args.max_target_length,
    )
    generator = torch.Generator().manual_seed(SEED)
    collate = make_collator(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=collate, pin_memory=use_cuda,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, pin_memory=use_cuda,
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not use_bf16)

    run_config = {
        **vars(args), "seed": SEED, "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "transformers_model_type": model.config.model_type,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if use_cuda else None,
        "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_cuda else "fp32"),
        "gradient_checkpointing": True,
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "train_pairs": len(train_en), "validation_pairs": len(dev_en), "test_pairs": len(test_en),
        "train_source_truncated": train_dataset.source_truncated,
        "train_target_truncated": train_dataset.target_truncated,
        "validation_source_truncated": dev_dataset.source_truncated,
        "validation_target_truncated": dev_dataset.target_truncated,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "optimizer": "AdamW", "scheduler": "linear_with_warmup",
        "warmup_steps": warmup_steps, "total_optimizer_steps": total_optimizer_steps,
        "decoding": {"method": "greedy", "beam_size": 1, "max_new_tokens": args.max_target_length},
    }
    save_json(run_dir / "run_config.json", run_config)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    save_json(run_dir / "status.json", {"state": "training", "started_at_utc": started_at, "epoch": 0})

    epoch_log = (run_dir / "epochs.jsonl").open("w", encoding="utf-8")
    step_log = (run_dir / "steps.jsonl").open("w", encoding="utf-8")
    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    global_micro_step = 0
    optimizer_step = 0
    stopped_reason = "epoch_cap"
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        batches = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_cuda):
                raw_loss = model(**batch).loss
                loss = raw_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            global_micro_step += 1
            loss_sum += float(raw_loss.detach())
            batches += 1
            update_now = batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader)
            if update_now:
                scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), 1.0))
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_was_skipped = scaler.get_scale() < scale_before
                if not step_was_skipped:
                    scheduler.step()
                    optimizer_step += 1
                    grad_norm_sum += grad_norm
                    grad_norm_count += 1
                if not step_was_skipped and optimizer_step % args.log_every == 0:
                    record = {
                        "optimizer_step": optimizer_step, "micro_step": global_micro_step,
                        "epoch": epoch, "batch": batch_index, "loss": float(raw_loss.detach()),
                        "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": grad_norm,
                        "elapsed_seconds": time.perf_counter() - training_started,
                        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated() if use_cuda else None,
                    }
                    step_log.write(json.dumps(record) + "\n")
                    step_log.flush()

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_cuda):
                    validation_loss += float(model(**batch).loss)
        train_loss = loss_sum / batches
        val_loss = validation_loss / len(dev_loader)
        dev_predictions, dev_seconds = timed_generate(
            lambda batch: generate(model, tokenizer, batch, args.max_source_length, args.max_target_length),
            dev_en, batch_size=args.batch_size,
        )
        dev_metrics = evaluate(dev_predictions, dev_hi)
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            model.save_pretrained(run_dir / "best_model", safe_serialization=True)
        else:
            stale_epochs += 1
        model.save_pretrained(run_dir / "last_model", safe_serialization=True)
        epoch_record = {
            "epoch": epoch, "optimizer_step": optimizer_step,
            "train_loss": train_loss, "validation_loss": val_loss,
            "train_perplexity": math.exp(min(train_loss, 20)),
            "validation_perplexity": math.exp(min(val_loss, 20)),
            "validation_bleu": dev_metrics["bleu"], "validation_chrf": dev_metrics["chrf"],
            "validation_inference_seconds": dev_seconds,
            "learning_rate": scheduler.get_last_lr()[0],
            "mean_gradient_norm": grad_norm_sum / max(1, grad_norm_count),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - training_started,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
            "improved": improved, "stale_epochs": stale_epochs,
        }
        history.append(epoch_record)
        epoch_log.write(json.dumps(epoch_record) + "\n")
        epoch_log.flush()
        save_json(run_dir / "status.json", {
            "state": "training", "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "stale_epochs": stale_epochs,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps(epoch_record), flush=True)
        if stale_epochs >= args.patience:
            stopped_reason = "early_stopping"
            break

    epoch_log.close()
    step_log.close()
    training_seconds = time.perf_counter() - training_started
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    model = AutoModelForSeq2SeqLM.from_pretrained(run_dir / "best_model").to(device)
    model.config.use_cache = True
    model.eval()
    save_json(run_dir / "status.json", {"state": "evaluating", "best_epoch": best_epoch, "stopped_reason": stopped_reason})
    predictions, inference_seconds = timed_generate(
        lambda batch: generate(model, tokenizer, batch, args.max_source_length, args.max_target_length),
        test_en, batch_size=args.batch_size,
    )
    test_metrics = evaluate(predictions, test_hi)
    (run_dir / "test_predictions.jsonl").write_text("".join(
        json.dumps({"id": i, "source": source, "reference": reference, "prediction": prediction}, ensure_ascii=False) + "\n"
        for i, (source, reference, prediction) in enumerate(zip(test_en, test_hi, predictions))
    ), encoding="utf-8")
    summary = {
        "run_id": args.run_id, "model_id": args.model_id, "stopped_reason": stopped_reason,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_test_sentence": inference_seconds / len(test_en),
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "test_bleu": test_metrics["bleu"], "test_chrf": test_metrics["chrf"],
        "test_sentences": len(test_en), "history": history,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {
        "state": "completed", "stopped_reason": stopped_reason,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"event": "training_completed", **summary}), flush=True)


if __name__ == "__main__":
    main()
