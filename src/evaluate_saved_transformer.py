"""Finish evaluation of a saved scratch-Transformer best checkpoint."""
import json
import time
from pathlib import Path

import torch
from torch import nn

from common import OUT, SEED, evaluate, save_json, split_data, timed_generate
from train_transformer import SentencePieceTokenizer, Transformer


def main():
    base = OUT / "transformer"
    logs = OUT / "logs" / "transformer"
    tokenizer_path = base / "tokenizer" / "iitb_en_hi.model"
    checkpoint_path = logs / "best_checkpoint.pt"
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(tokenizer.vocab_size, tokenizer.pad_token_id, d=config["d_model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _, _ = split_data("train")
    test_en, test_hi = split_data("test")
    def generate(batch):
        rows = [torch.tensor(tokenizer.encode(x, config["max_length"])) for x in batch]
        src = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        result = torch.full((len(batch), 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        done = torch.zeros(len(batch), dtype=torch.bool, device=device)
        for _ in range(config["max_length"]):
            nxt = model(src, result)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, tokenizer.pad_token_id), nxt)
            result = torch.cat([result, nxt[:, None]], dim=1)
            done |= nxt.eq(tokenizer.eos_token_id)
            if done.all(): break
        return tokenizer.batch_decode(result[:, 1:])
    predictions, elapsed = timed_generate(generate, test_en, batch_size=8)
    metrics = evaluate(predictions, test_hi)
    params = sum(p.numel() for p in model.parameters())
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "predictions" / "transformer_sentencepiece.jsonl").write_text(
        "".join(json.dumps({"source":s,"reference":r,"prediction":p}, ensure_ascii=False)+"\n" for s,r,p in zip(test_en,test_hi,predictions)), encoding="utf-8")
    summary = {"architecture":"scratch_transformer", "seed":SEED,
        "training_seconds":sum(x["epoch_seconds"] for x in (json.loads(l) for l in (logs / "epochs.jsonl").read_text(encoding="utf-8").splitlines())),
        "inference_seconds":elapsed,"seconds_per_sentence":elapsed/len(test_en),
        "total_parameters":params,"trainable_parameters":params,
        "best_validation_loss":checkpoint["best_validation_loss"],
        "train_pairs":30000,"validation_pairs":520,"test_pairs":len(test_en),
        "epochs":8,"model_config":config,**metrics}
    save_json(OUT / "metrics" / "transformer_sentencepiece.json",summary)
    save_json(logs / "final_summary.json",summary)
    torch.save({"state_dict":model.state_dict(),"config":config},base / "model_sentencepiece.pt")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
