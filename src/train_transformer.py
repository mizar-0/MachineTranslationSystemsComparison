"""Train a compact Transformer encoder-decoder from scratch on IITB EN-HI."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from common import OUT, SEED, evaluate, save_json, seed_everything, split_data, timed_generate


class ParallelDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, limit=128):
        self.sources, self.targets, self.tokenizer, self.limit = sources, targets, tokenizer, limit
    def __len__(self): return len(self.sources)
    def __getitem__(self, i):
        src = self.tokenizer(self.sources[i], max_length=self.limit, truncation=True, add_special_tokens=True)["input_ids"]
        tgt = self.tokenizer(self.targets[i], max_length=self.limit, truncation=True, add_special_tokens=True)["input_ids"]
        return torch.tensor(src), torch.tensor(tgt)


def collate(batch, pad):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad)
    return src, tgt


class Transformer(nn.Module):
    def __init__(self, vocab, pad, d=256, heads=4, layers=4, dropout=0.1, max_len=130):
        super().__init__()
        self.pad = pad
        self.embedding = nn.Embedding(vocab, d, padding_idx=pad)
        self.pos = nn.Embedding(max_len, d)
        self.transformer = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=layers,
                                          num_decoder_layers=layers, dim_feedforward=4*d,
                                          dropout=dropout, batch_first=True)
        self.output = nn.Linear(d, vocab)
    def emb(self, ids):
        p = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.embedding(ids) * math.sqrt(self.embedding.embedding_dim) + self.pos(p)
    def forward(self, src, tgt):
        length = tgt.shape[1]
        causal = torch.triu(torch.ones(length, length, device=tgt.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(self.emb(src), self.emb(tgt), tgt_mask=causal,
                             src_key_padding_mask=src.eq(self.pad),
                             tgt_key_padding_mask=tgt.eq(self.pad),
                             memory_key_padding_mask=src.eq(self.pad))
        return self.output(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/mt5-small", help="Tokenizer only; model weights are not used")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=256)
    args = ap.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train_en, train_hi = split_data("train")
    val_en, val_hi = split_data("dev")
    test_en, test_hi = split_data("test")
    train_en, train_hi = train_en[:args.max_train], train_hi[:args.max_train]
    pad, bos, eos = tok.pad_token_id, tok.pad_token_id, tok.eos_token_id
    train = DataLoader(ParallelDataset(train_en, train_hi, tok, args.max_length), batch_size=args.batch_size,
                       shuffle=True, collate_fn=lambda b: collate(b, pad))
    valid = DataLoader(ParallelDataset(val_en, val_hi, tok, args.max_length), batch_size=args.batch_size,
                       collate_fn=lambda b: collate(b, pad))
    model = Transformer(tok.vocab_size, pad, d=args.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    best, best_state, stale, patience = float("inf"), None, 0, 4
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for src, tgt in train:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item()
        model.eval(); vtotal = 0.0
        with torch.inference_mode():
            for src, tgt in valid:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt[:, :-1])
                vtotal += loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
        tr_loss, val_loss = total / len(train), vtotal / len(valid)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "validation_loss": val_loss})
        print(f"epoch {epoch+1}: train={tr_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state); model.eval()
    outdir = OUT / "transformer"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"vocab_size": tok.vocab_size,
               "pad_token_id": pad, "d_model": args.d_model, "tokenizer": args.tokenizer}}, outdir / "model.pt")
    tok.save_pretrained(outdir / "tokenizer")

    def generate(batch):
        encoded = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        src = encoded.input_ids
        result = torch.full((len(batch), 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(len(batch), dtype=torch.bool, device=device)
        for _ in range(args.max_length):
            logits = model(src, result)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            result = torch.cat([result, nxt[:, None]], dim=1)
            done |= nxt.eq(eos)
            if done.all(): break
        return tok.batch_decode(result[:, 1:], skip_special_tokens=True)
    preds, infer_seconds = timed_generate(generate, test_en, batch_size=8)
    metrics = evaluate(preds, test_hi)
    total_params = sum(p.numel() for p in model.parameters())
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "predictions" / "transformer.jsonl").write_text("".join(json.dumps({"source": s, "reference": r, "prediction": p}, ensure_ascii=False)+"\n" for s,r,p in zip(test_en,test_hi,preds)), encoding="utf-8")
    save_json(OUT / "metrics" / "transformer.json", {"architecture":"scratch_transformer","epochs":len(history),
        "training_seconds":train_seconds,"inference_seconds":infer_seconds,"seconds_per_sentence":infer_seconds/len(test_en),
        "total_parameters":total_params,"trainable_parameters":total_params,"best_validation_loss":best,
        "train_history":history,"seed":SEED,"train_pairs":len(train_en),"validation_pairs":len(val_en),"test_pairs":len(test_en),**metrics})


if __name__ == "__main__": main()
