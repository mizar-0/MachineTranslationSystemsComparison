"""Train and evaluate a reproducible Transformer encoder-decoder from scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import DATA, OUT, SEED, evaluate, save_json, seed_everything, split_data, timed_generate


class SentencePieceTokenizer:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_token_id, self.bos_token_id, self.eos_token_id = 0, 1, 2
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, limit: int) -> list[int]:
        return [self.bos_token_id] + self.sp.encode(text, out_type=int)[: limit - 2] + [self.eos_token_id]

    def batch_decode(self, rows) -> list[str]:
        return [self.sp.decode([int(i) for i in row if int(i) not in (0, 1, 2)]) for row in rows]


class ParallelDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, limit):
        self.rows = [(tokenizer.encode(s, limit), tokenizer.encode(t, limit)) for s, t in zip(sources, targets)]
        self.source_truncated = sum(len(tokenizer.sp.encode(s, out_type=int)) > limit - 2 for s in sources)
        self.target_truncated = sum(len(tokenizer.sp.encode(t, out_type=int)) > limit - 2 for t in targets)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, target = self.rows[index]
        return torch.tensor(source), torch.tensor(target)


def collate(rows, pad_id):
    sources, targets = zip(*rows)
    return (
        nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=pad_id),
        nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_id),
    )


class Transformer(nn.Module):
    def __init__(self, vocab, pad, d_model=256, heads=4, layers=4, dropout=0.1, max_len=130):
        super().__init__()
        self.pad = pad
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.position = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=heads, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.output = nn.Linear(d_model, vocab)

    def embed(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.embedding(ids) * math.sqrt(self.embedding.embedding_dim) + self.position(positions)

    def forward(self, source, target):
        length = target.shape[1]
        causal = torch.triu(torch.ones(length, length, device=target.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(
            self.embed(source), self.embed(target), tgt_mask=causal,
            src_key_padding_mask=source.eq(self.pad),
            tgt_key_padding_mask=target.eq(self.pad),
            memory_key_padding_mask=source.eq(self.pad),
        )
        return self.output(hidden)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def greedy_translate(model, tokenizer, texts, max_length, device):
    rows = [torch.tensor(tokenizer.encode(text, max_length)) for text in texts]
    source = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    result = torch.full((len(texts), 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(len(texts), dtype=torch.bool, device=device)
    for _ in range(max_length):
        next_ids = model(source, result)[:, -1].argmax(-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, tokenizer.pad_token_id), next_ids)
        result = torch.cat([result, next_ids[:, None]], dim=1)
        finished |= next_ids.eq(tokenizer.eos_token_id)
        if finished.all():
            break
    return tokenizer.batch_decode(result[:, 1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = OUT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_dir = run_dir / "tokenizer"
    tokenizer_dir.mkdir()
    status_path = run_dir / "status.json"
    started_at = datetime.now(timezone.utc).isoformat()
    save_json(status_path, {"state": "initializing", "started_at_utc": started_at})

    train_en, train_hi = split_data("train")
    dev_en, dev_hi = split_data("dev")
    test_en, test_hi = split_data("test")
    train_en, train_hi = train_en[: args.max_train], train_hi[: args.max_train]

    corpus_path = run_dir / "tokenizer_training_corpus.txt"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as corpus:
        for english, hindi in zip(train_en, train_hi):
            corpus.write(english.replace("\n", " ") + "\n")
            corpus.write(hindi.replace("\n", " ") + "\n")
    model_prefix = tokenizer_dir / "iitb_en_hi"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path), model_prefix=str(model_prefix), vocab_size=args.vocab_size,
        model_type="bpe", character_coverage=0.9995, pad_id=0, bos_id=1,
        eos_id=2, unk_id=3, shuffle_input_sentence=False, input_sentence_size=0,
        num_threads=8, minloglevel=1,
    )
    tokenizer = SentencePieceTokenizer(model_prefix.with_suffix(".model"))
    pad_id = tokenizer.pad_token_id

    train_dataset = ParallelDataset(train_en, train_hi, tokenizer, args.max_length)
    dev_dataset = ParallelDataset(dev_en, dev_hi, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        collate_fn=lambda rows: collate(rows, pad_id), pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda rows: collate(rows, pad_id), pin_memory=torch.cuda.is_available(),
    )

    model = Transformer(
        tokenizer.vocab_size, pad_id, args.d_model, args.heads,
        args.layers, args.dropout, args.max_length + 2,
    ).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DATA.parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_commit = None
    run_config = {
        **vars(args), "seed": SEED, "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "total_parameters": total_parameters, "train_pairs": len(train_en),
        "validation_pairs": len(dev_en), "test_pairs": len(test_en),
        "tokenizer": "SentencePiece BPE trained only on the selected training split",
        "tokenizer_model_sha256": sha256(model_prefix.with_suffix(".model")),
        "train_source_truncated": train_dataset.source_truncated,
        "train_target_truncated": train_dataset.target_truncated,
        "validation_source_truncated": dev_dataset.source_truncated,
        "validation_target_truncated": dev_dataset.target_truncated,
        "dataset_manifest_sha256": sha256(DATA / "manifest.json"),
        "git_commit": git_commit,
        "decoding": {"method": "greedy", "beam_size": 1, "max_new_tokens": args.max_length},
    }
    save_json(run_dir / "run_config.json", run_config)
    save_json(status_path, {"state": "training", "started_at_utc": started_at, "epoch": 0})

    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    global_step = 0
    stopped_reason = "epoch_cap"
    training_started = time.perf_counter()
    epoch_log = (run_dir / "epochs.jsonl").open("w", encoding="utf-8")
    step_log = (run_dir / "steps.jsonl").open("w", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        training_loss = 0.0
        training_tokens = 0
        grad_norm_total = 0.0
        for batch_index, (source, target) in enumerate(train_loader, start=1):
            source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            global_step += 1
            token_count = int(target[:, 1:].ne(pad_id).sum())
            training_loss += float(loss) * token_count
            training_tokens += token_count
            grad_norm_total += grad_norm
            if global_step % args.log_every == 0:
                record = {
                    "global_step": global_step, "epoch": epoch, "batch": batch_index,
                    "loss": float(loss), "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm,
                    "elapsed_seconds": time.perf_counter() - training_started,
                    "gpu_memory_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                }
                step_log.write(json.dumps(record) + "\n")
                step_log.flush()

        model.eval()
        validation_loss = 0.0
        validation_tokens = 0
        with torch.inference_mode():
            for source, target in dev_loader:
                source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
                logits = model(source, target[:, :-1])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
                token_count = int(target[:, 1:].ne(pad_id).sum())
                validation_loss += float(loss) * token_count
                validation_tokens += token_count
        train_loss = training_loss / training_tokens
        val_loss = validation_loss / validation_tokens
        scheduler.step(val_loss)

        dev_predictions, dev_seconds = timed_generate(
            lambda batch: greedy_translate(model, tokenizer, batch, args.max_length, device),
            dev_en, batch_size=8,
        )
        dev_metrics = evaluate(dev_predictions, dev_hi)
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save({
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_validation_loss": best_loss, "config": run_config,
            }, run_dir / "best_checkpoint.pt")
        else:
            stale_epochs += 1
        torch.save({
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_validation_loss": best_loss, "config": run_config,
        }, run_dir / "last_checkpoint.pt")

        epoch_record = {
            "epoch": epoch, "global_step": global_step,
            "train_loss": train_loss, "validation_loss": val_loss,
            "train_perplexity": math.exp(min(train_loss, 20)),
            "validation_perplexity": math.exp(min(val_loss, 20)),
            "validation_bleu": dev_metrics["bleu"], "validation_chrf": dev_metrics["chrf"],
            "validation_inference_seconds": dev_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mean_gradient_norm": grad_norm_total / len(train_loader),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - training_started,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "improved": improved, "stale_epochs": stale_epochs,
        }
        history.append(epoch_record)
        epoch_log.write(json.dumps(epoch_record) + "\n")
        epoch_log.flush()
        save_json(status_path, {
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
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    save_json(status_path, {"state": "evaluating", "best_epoch": best_epoch, "stopped_reason": stopped_reason})
    test_predictions, inference_seconds = timed_generate(
        lambda batch: greedy_translate(model, tokenizer, batch, args.max_length, device),
        test_en, batch_size=8,
    )
    test_metrics = evaluate(test_predictions, test_hi)
    (run_dir / "test_predictions.jsonl").write_text("".join(
        json.dumps({"id": i, "source": source, "reference": reference, "prediction": prediction}, ensure_ascii=False) + "\n"
        for i, (source, reference, prediction) in enumerate(zip(test_en, test_hi, test_predictions))
    ), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "config": run_config}, run_dir / "final_model.pt")
    summary = {
        "run_id": args.run_id, "stopped_reason": stopped_reason,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_test_sentence": inference_seconds / len(test_en),
        "total_parameters": total_parameters, "trainable_parameters": total_parameters,
        "test_bleu": test_metrics["bleu"], "test_chrf": test_metrics["chrf"],
        "test_sentences": len(test_en), "history": history,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(status_path, {
        "state": "completed", "stopped_reason": stopped_reason,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"event": "training_completed", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
