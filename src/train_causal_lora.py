"""LoRA-adapt a causal language model for English-to-Hindi translation.

The trainer deliberately avoids Trainer/datasets/pandas so it works in the
restricted Windows environment used for this project.  Every run is isolated
under outputs/runs and records the information needed by the final report.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import OUT, SEED, evaluate, save_json, seed_everything, split_data, timed_generate


DEFAULT_MODELS = {
    # These mirrors expose the original full-precision weights without requiring
    # interactive gated-license authentication. Exact IDs/revisions are logged.
    "llama": "unsloth/Llama-3.2-1B",
    "gemma": "unsloth/gemma-3-270m-it",
}


def prompt_for(source: str) -> str:
    return f"Translate the following English sentence into Hindi.\nEnglish: {source}\nHindi:"


def configure_greedy_generation(model):
    """Remove sampling defaults inherited from instruction-model configs."""
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


class CausalTranslationDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_source_length, max_target_length):
        self.rows = []
        self.source_truncated = 0
        self.target_truncated = 0
        eos_id = tokenizer.eos_token_id
        for source, target in zip(sources, targets):
            prompt_ids_full = tokenizer(prompt_for(source), add_special_tokens=True)["input_ids"]
            target_ids_full = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
            self.source_truncated += len(prompt_ids_full) > max_source_length
            self.target_truncated += len(target_ids_full) + int(eos_id is not None) > max_target_length
            prompt_ids = prompt_ids_full[:max_source_length]
            reserve_eos = 1 if eos_id is not None else 0
            target_ids = target_ids_full[: max_target_length - reserve_eos]
            if eos_id is not None:
                target_ids = target_ids + [eos_id]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            self.rows.append((input_ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(tokenizer):
    def collate(rows):
        max_length = max(len(input_ids) for input_ids, _ in rows)
        input_batch, attention_batch, label_batch = [], [], []
        for input_ids, labels in rows:
            padding = max_length - len(input_ids)
            input_batch.append(input_ids + [tokenizer.pad_token_id] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
            label_batch.append(labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
        }
    return collate


def generate(model, tokenizer, sources, max_source_length, max_new_tokens):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [prompt_for(source) for source in sources], padding=True, truncation=True,
        max_length=max_source_length, return_tensors="pt",
    ).to(model.device)
    tokenizer.padding_side = old_padding_side
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return [text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id")
    parser.add_argument(
        "--model-path",
        help="Optional local snapshot path; the source model ID is still recorded",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-source-length", type=int, default=96)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    seed_everything()
    model_id = args.model_id or DEFAULT_MODELS[args.model]
    model_load_path = args.model_path or model_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    run_dir = OUT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    save_json(run_dir / "status.json", {
        "state": "downloading_model", "started_at_utc": started_at,
        "model": args.model, "model_id": model_id,
        "model_load_path": model_load_path,
    })

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    configure_greedy_generation(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
    ))
    model.to(device)

    train_en, train_hi = split_data("train")
    dev_en, dev_hi = split_data("dev")
    test_en, test_hi = split_data("test")
    train_en, train_hi = train_en[:args.max_train], train_hi[:args.max_train]
    train_dataset = CausalTranslationDataset(
        train_en, train_hi, tokenizer, args.max_source_length, args.max_target_length,
    )
    dev_dataset = CausalTranslationDataset(
        dev_en, dev_hi, tokenizer, args.max_source_length, args.max_target_length,
    )
    collate = make_collator(tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=collate, pin_memory=use_cuda,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.validation_batch_size, shuffle=False,
        collate_fn=collate, pin_memory=use_cuda,
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=0.01,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not use_bf16)

    run_config = {
        **vars(args), "model_id": model_id, "model_load_path": model_load_path,
        "seed": SEED, "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers_model_type": model.config.model_type,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if use_cuda else None,
        "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_cuda else "fp32"),
        "gradient_checkpointing": True,
        "adaptation": "LoRA", "lora_target_modules": "all-linear",
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "trainable_parameter_percent": 100 * trainable_parameters / total_parameters,
        "train_pairs": len(train_en), "validation_pairs": len(dev_en), "test_pairs": len(test_en),
        "train_source_truncated": train_dataset.source_truncated,
        "train_target_truncated": train_dataset.target_truncated,
        "validation_source_truncated": dev_dataset.source_truncated,
        "validation_target_truncated": dev_dataset.target_truncated,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "optimizer": "AdamW", "scheduler": "linear_with_warmup",
        "warmup_steps": warmup_steps, "total_optimizer_steps": total_optimizer_steps,
        "prompt_template": prompt_for("{source}"),
        "label_masking": "prompt and padding tokens use label -100",
        "decoding": {"method": "greedy", "beam_size": 1, "max_new_tokens": args.max_target_length},
    }
    save_json(run_dir / "run_config.json", run_config)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    save_json(run_dir / "status.json", {
        "state": "training", "started_at_utc": started_at, "epoch": 0,
        "optimizer_step": 0, "total_optimizer_steps": total_optimizer_steps,
    })

    epoch_log = (run_dir / "epochs.jsonl").open("w", encoding="utf-8")
    step_log = (run_dir / "steps.jsonl").open("w", encoding="utf-8")
    history = []
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    global_micro_step = optimizer_step = 0
    stopped_reason = "epoch_cap"
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = grad_norm_sum = 0.0
        batches = grad_norm_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_cuda):
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
                        "total_optimizer_steps": total_optimizer_steps,
                        "epoch": epoch, "batch": batch_index,
                        "loss": float(raw_loss.detach()),
                        "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": grad_norm,
                        "elapsed_seconds": time.perf_counter() - training_started,
                        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated() if use_cuda else None,
                    }
                    step_log.write(json.dumps(record) + "\n")
                    step_log.flush()
                    save_json(run_dir / "status.json", {
                        "state": "training", "epoch": epoch, "batch": batch_index,
                        "optimizer_step": optimizer_step,
                        "total_optimizer_steps": total_optimizer_steps,
                        "best_epoch": best_epoch,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    })

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_cuda):
                    validation_loss += float(model(**batch).loss)
        train_loss = loss_sum / batches
        val_loss = validation_loss / len(dev_loader)
        # Preserve the completed epoch before generation-based metrics. This
        # makes the expensive training work recoverable even if decoding or a
        # metric library fails afterwards.
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            model.save_pretrained(run_dir / "best_adapter", safe_serialization=True)
        else:
            stale_epochs += 1
        model.save_pretrained(run_dir / "last_adapter", safe_serialization=True)
        save_json(run_dir / "epoch_checkpoint.json", {
            "epoch": epoch, "optimizer_step": optimizer_step,
            "train_loss": train_loss, "validation_loss": val_loss,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "stale_epochs": stale_epochs,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        dev_predictions, dev_seconds = timed_generate(
            lambda items: generate(
                model, tokenizer, items, args.max_source_length, args.max_target_length,
            ), dev_en, batch_size=args.generation_batch_size,
        )
        dev_metrics = evaluate(dev_predictions, dev_hi)
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
            "state": "training", "epoch": epoch, "optimizer_step": optimizer_step,
            "total_optimizer_steps": total_optimizer_steps,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "stale_epochs": stale_epochs,
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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_load_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, run_dir / "best_adapter").to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    configure_greedy_generation(model)
    model.eval()
    save_json(run_dir / "status.json", {
        "state": "evaluating", "best_epoch": best_epoch,
        "stopped_reason": stopped_reason,
    })
    predictions, inference_seconds = timed_generate(
        lambda items: generate(
            model, tokenizer, items, args.max_source_length, args.max_target_length,
        ), test_en, batch_size=args.generation_batch_size,
    )
    test_metrics = evaluate(predictions, test_hi)
    (run_dir / "test_predictions.jsonl").write_text("".join(
        json.dumps({
            "id": index, "source": source, "reference": reference,
            "prediction": prediction,
        }, ensure_ascii=False) + "\n"
        for index, (source, reference, prediction) in enumerate(zip(test_en, test_hi, predictions))
    ), encoding="utf-8")
    summary = {
        "run_id": args.run_id, "model_family": args.model, "model_id": model_id,
        "adaptation": "LoRA", "stopped_reason": stopped_reason,
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
