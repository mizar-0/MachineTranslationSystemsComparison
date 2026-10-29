"""Fine-tune mT5, LLaMA, or Gemma with one shared translation protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import DATA, OUT, SEED, count_parameters, evaluate, save_json, seed_everything, split_data, timed_generate

MODELS = {
    "mt5": ("google/mt5-small", "seq2seq"),
    "llama": ("meta-llama/Llama-3.2-1B", "causal"),
    "gemma": ("google/gemma-3-1b-it", "causal"),
}


class EncodedPairs(Dataset):
    def __init__(self, sources, targets, tokenizer, architecture, max_length):
        self.rows = []
        for source, target in zip(sources, targets):
            prompt = f"Translate English to Hindi: {source}\nHindi:" if architecture == "causal" else f"translate English to Hindi: {source}"
            if architecture == "causal":
                complete = tokenizer(prompt + " " + target, truncation=True, max_length=max_length)
                prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
                labels = complete["input_ids"].copy()
                labels[:len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
            else:
                complete = tokenizer(prompt, truncation=True, max_length=max_length)
                labels = tokenizer(text_target=target, truncation=True, max_length=max_length)["input_ids"]
            self.rows.append((complete["input_ids"], labels))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def make_collator(tokenizer):
    def collate(rows):
        max_input = max(len(x[0]) for x in rows)
        max_label = max(len(x[1]) for x in rows)
        ids, masks, labels = [], [], []
        for input_ids, label_ids in rows:
            ids.append(input_ids + [tokenizer.pad_token_id] * (max_input - len(input_ids)))
            masks.append([1] * len(input_ids) + [0] * (max_input - len(input_ids)))
            labels.append(label_ids + [-100] * (max_label - len(label_ids)))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}
    return collate


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
        load_kwargs["device_map"] = "auto"
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    if architecture == "causal":
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        ))

    train_ds = EncodedPairs(train_en, train_hi, tokenizer, architecture, args.max_length)
    val_ds = EncodedPairs(val_en, val_hi, tokenizer, architecture, args.max_length)
    collate = make_collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4 if architecture == "causal" else 5e-4)
    best, best_state, history, stale = float("inf"), None, [], 0
    start = time.perf_counter()
    for epoch in range(int(args.epochs)):
        model.train(); total = 0.0
        order = torch.randperm(len(train_en)).tolist()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * args.grad_accum
        model.eval(); val_total = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                val_total += model(**batch).loss.item()
        val_loss = val_total / max(1, len(val_loader))
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, len(train_loader)), "validation_loss": val_loss})
        print(f"epoch {epoch+1}: validation_loss={val_loss:.4f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.save_pretrained(str(output / "best"))
        else:
            stale += 1
            if stale >= 2: break
    train_seconds = time.perf_counter() - start
    if best_state is not None: model.load_state_dict(best_state)
    model.save_pretrained(str(output / "model"))
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
        "best_validation_loss": best, "train_history": history,
        **metrics,
    })
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
