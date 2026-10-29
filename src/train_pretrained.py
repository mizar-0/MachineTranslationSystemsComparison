"""Fine-tune mT5, LLaMA, or Gemma with one shared translation protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import DATA, OUT, SEED, count_parameters, evaluate, save_json, seed_everything, split_data, timed_generate

MODELS = {
    "mt5": ("google/mt5-small", "seq2seq"),
    "llama": ("meta-llama/Llama-3.2-1B", "causal"),
    "gemma": ("google/gemma-3-1b-it", "causal"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-id", help="Override the default Hugging Face model id")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    seed_everything()
    model_id, architecture = MODELS[args.model]
    model_id = args.model_id or model_id
    output = OUT / args.model
    train_en, train_hi = split_data("train")
    val_en, val_hi = split_data("dev")
    test_en, test_hi = split_data("test")
    train_en, train_hi = train_en[: args.max_train], train_hi[: args.max_train]

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForSeq2SeqLM if architecture == "seq2seq" else AutoModelForCausalLM
    load_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if torch.cuda.is_available() and architecture == "causal":
        try:
            load_kwargs.update(load_in_4bit=True, device_map="auto")
        except Exception:
            pass
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    if architecture == "causal":
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        ))

    def encode_batch(sources: list[str], targets: list[str], causal: bool):
        prompts = [f"Translate English to Hindi: {s}\nHindi:" for s in sources] if causal else [f"translate English to Hindi: {s}" for s in sources]
        encoded = tokenizer(prompts, max_length=args.max_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_length, truncation=True)
        encoded["labels"] = labels["input_ids"]
        if causal:
            # The trainer uses tokenized target Hindi as labels. Mask source tokens.
            for row, source_len in zip(encoded["labels"], [len(x) for x in encoded["input_ids"]]):
                row[:source_len] = [-100] * min(source_len, len(row))
        return encoded

    def dataset(sources, targets):
        rows = encode_batch(sources, targets, architecture == "causal")
        return Dataset.from_dict(rows)

    train_ds = dataset(train_en, train_hi)
    val_ds = dataset(val_en, val_hi)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4 if architecture == "causal" else 5e-4,
        fp16=torch.cuda.is_available(), gradient_checkpointing=architecture == "causal",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=100, report_to="none", seed=SEED,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    start = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - start
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(str(output / "model"))
    model.config.use_cache = True
    model.eval()

    def generate(batch):
        prompts = [f"Translate English to Hindi: {s}\nHindi:" for s in batch] if architecture == "causal" else [f"translate English to Hindi: {s}" for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(model.device)
        kwargs = dict(max_new_tokens=128, do_sample=args.temperature > 0, pad_token_id=tokenizer.pad_token_id)
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **kwargs)
        if architecture == "causal":
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    predictions, inference_seconds = timed_generate(generate, test_en, args.batch_size)
    metrics = evaluate(predictions, test_hi)
    total, trainable = count_parameters(model)
    from common import OUT
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "predictions" / f"{args.model}.jsonl").write_text(
        "".join(json.dumps({"source": s, "reference": r, "prediction": p}, ensure_ascii=False) + "\n" for s, r, p in zip(test_en, test_hi, predictions)), encoding="utf-8")
    save_json(OUT / "metrics" / f"{args.model}.json", {
        "model_id": model_id, "architecture": architecture, "train_pairs": len(train_en),
        "validation_pairs": len(val_en), "test_pairs": len(test_en), "seed": SEED,
        "training_seconds": train_seconds, "inference_seconds": inference_seconds,
        "seconds_per_sentence": inference_seconds / len(test_en),
        "total_parameters": total, "trainable_parameters": trainable,
        "best_validation_loss": trainer.state.best_metric,
        "train_loss": result.training_loss, **metrics,
    })
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
