# EXP-001: SP2048 setup

## Why

Published challenge shards are **sp1024 only**. EXP-001 needs matching `fineweb10B_sp2048` bins + `fineweb_2048_bpe.model`.

## One-time build (full export, slow)

```bash
pip install sentencepiece huggingface_hub numpy
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data/sp2048_full \
  --tokenizer-config data/tokenizer_specs_sp2048.json
mkdir -p data/datasets data/tokenizers
ln -sfn "$(pwd)/data/sp2048_full/datasets/fineweb10B_sp2048" data/datasets/fineweb10B_sp2048
cp data/sp2048_full/tokenizers/fineweb_2048_bpe.model data/tokenizers/
```

## Fast proxy (same val prefix, truncated train)

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data/sp2048_proxy \
  --tokenizer-config data/tokenizer_specs_sp2048.json \
  --tokenizer-train-docs 500000 \
  --proxy-max-docs 600000
```

Then symlink/copy as above from `sp2048_proxy`.

## Run EXP-001

```bash
ITERATIONS=200 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_exp001.py
```

Parse last `val_bpb:` before `final_int8_zlib_roundtrip` (or use roundtrip line). Keep if `val_bpb < best - 0.002`.
