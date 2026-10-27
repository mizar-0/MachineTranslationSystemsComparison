# Machine Translation Systems Comparison

This repository contains the code for comparing four English-to-Hindi machine
translation systems:

- a Transformer encoder-decoder trained from scratch;
- mT5 fine-tuning;
- LLaMA adaptation;
- Gemma adaptation.

## Dataset protocol

The project uses the official IIT Bombay English-Hindi Parallel Corpus files
stored one directory above this repository. The corpus is treated as already
prepared and trusted. The subset procedure performs no normalization,
deduplication, length filtering, language detection, or other text cleaning.

The fixed experiment splits are:

- 30,000 randomly sampled pairs from the official training files;
- all 520 pairs from the official development files;
- all 2,507 pairs from the official test files.

The training subset uses reservoir sampling with seed `1729`. The generated
`train_indices.txt` and `manifest.json` record the exact selection and output
checksums.

## Create the local dataset subset

From the repository root, run:

```powershell
python scripts/prepare_iitb_subset.py
```

The generated files are written to `data/iitb_en_hi_30k/`. Dataset files and
future model artifacts are ignored by Git; the reproducible preparation script
is tracked.
