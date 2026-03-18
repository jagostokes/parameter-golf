# `train_gpt_new.py` ablation log

## BPB definition (sanity check)

The challenge uses:

\[
\text{val\_bpb} = \frac{\text{CE (nats)}}{\ln 2} \times \frac{\text{\#tokens}}{\text{\#raw UTF-8 bytes}}
\]

Training minimizes token CE; a better tokenizer increases bytes per token (fewer tokens per byte), which **lowers** bpb for the same per-token loss. The script logs `bpb_formula_check` alongside `val_bpb` each validation.

## Smoke test (200 steps)

**With sp1024 (always works after repo data download):**
```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 ITERATIONS=200 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_new.py
```

**With sp8192 (if available on hub):**
```bash
python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1
DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 ITERATIONS=200 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_new.py
```

Default `NUM_RECURSE=10` keeps steps fast. Target: `val_bpb < 1.2244` after full training (200 iters is only a smoke run).

### Estimate 8× H100 10‑min run on 1 GPU

To approximate the **step count** (and thus val_bpb) you’d get on 8× H100 in 10 minutes, run the same number of steps on Colab/1 GPU. Baseline did **13,780** steps in 10 min; this model (10 recurrences) is in the same ballpark, so use **10,000 steps** as a round estimate.

Set **`ESTIMATE_8H100_STEPS=10000`** and **`MAX_WALLCLOCK_SECONDS=0`** (no wall cap). At ~1 s/step that’s **~2.8 h** on 1 GPU. When `ESTIMATE_8H100_STEPS` is set, **validation runs every 500 steps** by default so you can track val_bpb (leaderboard target &lt; 1.2244). Override with `VAL_LOSS_EVERY=0` to validate only at the end.

```bash
ESTIMATE_8H100_STEPS=10000 MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 VAL_LOSS_EVERY=500 TRAIN_BATCH_TOKENS=65536 \
torchrun --standalone --nproc_per_node=1 train_gpt_new.py
```

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

## Optional tuning (both recommended)

### 1. Larger vocab (8192)

Better tokenizer → fewer tokens per byte → lower bpb for same loss. Use **matching** data and tokenizer.

- **If the hub has sp8192:**
  ```bash
  python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
  ```
  Then:
  ```bash
  DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  torchrun --standalone --nproc_per_node=8 train_gpt_new.py
  ```
- **If only sp4096 exists:** use `--variant sp4096`, `VOCAB_SIZE=4096`, and the tokenizer path from the download (e.g. `fineweb_4096_bpe.model`).
- **Otherwise:** keep `sp1024` + `VOCAB_SIZE=1024` (default data from repo).

### 2. NUM_RECURSE (10 vs 20)

- **Default is now 10** – faster steps, often similar or better bpb when you can run more steps in the same time.
- For a **full 10‑min run** on 8× H100, try **20** for maximum depth:
  ```bash
  NUM_RECURSE=20 torchrun --standalone --nproc_per_node=8 train_gpt_new.py
  ```
- Ablate: run the same iteration count with `NUM_RECURSE=10` and `NUM_RECURSE=20`, compare val_bpb and time.

## Size fallback

If `assert compressed_bytes < 16_000_000` fails with `VOCAB_SIZE=8192`, reduce to `4096`, re-download `fineweb10B_sp4096`, and retrain tokenizer (`RETRAIN_TOKENIZER=1` + `TOKENIZER_TRAIN_INPUT` on FineWeb text, or hub shard variant).

## Env reference

| Variable | Role |
|----------|------|
| `DATA_PATH` | Shard directory (must match tokenizer) |
| `TOKENIZER_PATH` | SentencePiece `.model` |
| `VOCAB_SIZE` | Must match SP vocab |
| `ITERATIONS` | Max training steps (default 20000). Overridden by `ESTIMATE_8H100_STEPS` if set. |
| `ESTIMATE_8H100_STEPS` | If set (e.g. 10000), run this many steps to approximate 8× H100 10‑min run on 1 GPU. |
| `VAL_LOSS_EVERY` | Validate every N steps (0 = only at end). Defaults to 500 when `ESTIMATE_8H100_STEPS` is set. |
| `NUM_RECURSE` | Recurrence depth (default 10; use 20 for full runs) |
| `MLA_RANK` | KV bottleneck rank (default 64) |
| `WINDOW_SHORT` / `WINDOW_LONG` | Alternating Flex windows per step |
| `QAT` | `1` = int8 STE on matmul weights |
| `RETRAIN_TOKENIZER` | `1` trains SP from `TOKENIZER_TRAIN_INPUT` |
| `COMPILE_MODE` | `max-autotune` (or `reduce-overhead` if OOM) |
