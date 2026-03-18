# `train_gpt_new.py` ablation log

## BPB definition (sanity check)

The challenge uses:

\[
\text{val\_bpb} = \frac{\text{CE (nats)}}{\ln 2} \times \frac{\text{\#tokens}}{\text{\#raw UTF-8 bytes}}
\]

Training minimizes token CE; a better tokenizer increases bytes per token (fewer tokens per byte), which **lowers** bpb for the same per-token loss. The script logs `bpb_formula_check` alongside `val_bpb` each validation.

## Smoke test (200 steps)

```bash
# Requires dataset + tokenizer with matching vocab (e.g. sp8192 or sp4096 from HF manifest).
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/<matching_8192_bpe.model> \
VOCAB_SIZE=8192 \
ITERATIONS=200 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_new.py
```

Target: `val_bpb < 1.2244` after full training (200 iters is only a smoke run; expect higher bpb until converged).

## Ablation table (fill in after experiments)

| Variant | val_bpb | Δ vs full | Notes |
|---------|---------|-----------|-------|
| Full stack | | — | |
| −8192 tok (keep 1024 data) | | | invalid pairing — always match DATA_PATH to tokenizer |
| −MLA (full KV) | | | |
| −Flex (full causal SDPA) | | | |
| −QAT | | | `QAT=0` |
| −Smear | | | |
| −U-net skips | | | |
| −step_embed | | | |
| −Muon (AdamW all) | | | |
| −low-rank linear (dense init) | | | |
| −partial RoPE | | | |
| −value embedding path | | | |

Remove or gate features that **hurt** bpb on your hardware/budget.

## Size fallback

If `assert compressed_bytes < 16_000_000` fails with `VOCAB_SIZE=8192`, reduce to `4096`, re-download `fineweb10B_sp4096`, and retrain tokenizer (`RETRAIN_TOKENIZER=1` + `TOKENIZER_TRAIN_INPUT` on FineWeb text, or hub shard variant).

## Env reference

| Variable | Role |
|----------|------|
| `DATA_PATH` | Shard directory (must match tokenizer) |
| `TOKENIZER_PATH` | SentencePiece `.model` |
| `VOCAB_SIZE` | Must match SP vocab |
| `NUM_RECURSE` | Universal block depth (default 20) |
| `MLA_RANK` | KV bottleneck rank (default 64) |
| `WINDOW_SHORT` / `WINDOW_LONG` | Alternating Flex windows per step |
| `QAT` | `1` = int8 STE on matmul weights |
| `RETRAIN_TOKENIZER` | `1` trains SP from `TOKENIZER_TRAIN_INPUT` |
| `COMPILE_MODE` | `max-autotune` (or `reduce-overhead` if OOM) |
