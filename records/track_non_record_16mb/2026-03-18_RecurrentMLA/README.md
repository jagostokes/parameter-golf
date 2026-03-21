# Recurrent MLA (non-record submission)

Depth-recurrent transformer with a **single shared block** applied `NUM_RECURSE` times (Universal Transformer style), MLA-style low-rank KV, Muon optimizer, QAT, and optional FlexAttention. Non-record: does not meet the 10-minute 8× H100 leaderboard cutoff; submitted for the 16 MB artifact cap and reproducibility.

## Architecture

- **Recurrence:** One shared `MLARecurrentBlock` (attention + MLP) applied 10× (default) with learned step embeddings and U-net skips at steps N/2 and N.
- **Attention:** MLA (multi-head latent attention): KV projected to rank 64 then back up; Q full; partial RoPE (50% of head dims); QK-norm; optional FlexAttention sliding windows (512 / 2048).
- **MLP:** ReLU², RMSNorm, no bias; value embeddings (Zhou et al.) mixed into attention output.
- **Embedding:** Smear module (gate on first 12 dims, causal mix with previous position).
- **Optimizer:** Muon for 2D weights (Newton–Schulz orthogonalization); AdamW for embeddings and scalars. Linear warmup + cosine decay + 20% linear cooldown.
- **QAT:** Optional int8 fake-quant with STE from step 1. Low-rank linear init (W = A @ B, muP scaling).

## Configuration

- **Layout:** `VOCAB_SIZE=1024` (or 8192 with matching data), `MODEL_DIM=512`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `NUM_RECURSE=10`, `MLA_RANK=64`, tied embeddings.
- **Batching:** `TRAIN_BATCH_TOKENS=524288` (or 65536 for 1-GPU), `TRAIN_SEQ_LEN=1024`.
- **Validation:** Full `fineweb_val_*`; when `ESTIMATE_8H100_STEPS` is set, every 500 steps by default.

## Command (1-GPU estimate run, 10k steps)

```bash
ESTIMATE_8H100_STEPS=10000 MAX_WALLCLOCK_SECONDS=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=65536 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Command (8× H100, 10-min cap)

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 COMPILE_MODE=max-autotune \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Metrics (from submitted run)

See `train.log` and `submission.json`. Replace with final numbers after your 10k-step or 8× H100 run.

## Included files

- `train_gpt.py` – training script (snapshot)
- `train.log` – training log (replace with full run log if needed)
- `submission.json` – leaderboard metadata
- `README.md` – this file
