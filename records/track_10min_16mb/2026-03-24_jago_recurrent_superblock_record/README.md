# Recurrent superblock (non-TTT) record attempt

**Author:** jago  
**Date:** 2026-03-24  
**Status:** record attempt — run logs and metrics to be filled after 8×H100 training.

## Thesis

Leaderboard gains are increasingly marginal on fixed **independent-layer** backbones under a **16MB** cap. This submission trades **unique weight matrices** for **depth**: six **physical** transformer blocks are each **run twice** (12 **virtual** layers) with **shared** attention and MLP weights, while **per-virtual** RMSNorm weights, residual gates, and **rank-8 depth adapters** keep virtual layers distinguishable. That frees bytes for wider `d_model` / MLP ratio and a stronger eval stack (**sliding-window BPB**, **EMA**, **aligned STE QAT** into **mixed int5/int6** + **LZMA**), **without TTT**.

## Architecture (summary)

| Component | Setting |
|-----------|---------|
| Virtual layers | 12 (`VIRTUAL_LAYERS=12`) |
| Physical shared blocks | 6 (`PHYSICAL_LAYERS=6`) — virtual `v` uses physical `v // 2` weights |
| `d_model` | 576 default (`MODEL_DIM`); try 640 if artifact budget allows |
| Heads / GQA | 8 heads, 4 KV |
| MLP | `MLP_MULT=3.5`, **LeakyReLU(0.5)²** |
| XSA | Last `XSA_LAST_N` virtual layers (default **12** = all) |
| RoPE | Partial: `PARTIAL_ROPE_DIM=16` on head prefix |
| BigramHash | `BIGRAM_HASH=2048` (set `0` to disable) |
| Value residual / gated attn | `VR_ENABLED`, `GA_ENABLED` |
| U-Net | 6 virtual encoder + 6 virtual decoder + skip weights |
| Virtual depth dropout | Random **8–10** virtual layers early; full **12** in last **25%** of iterations (`VDD_END_FULL_FRAC=0.25`) |
| Optimizer | **Muon** on 3D banks + **AdamW** on embeddings / scalars / small matrices |
| EMA | `EMA_ENABLED=1`, decay `0.997` |
| QAT | Rowwise fake-quant (6-bit) on banks after step ≥ `QAT_START_FRAC × iterations` |
| Export | Per-row **int6** if sensitivity ≥ median else **int5**; **LZMA** (or `USE_ZSTD_FALLBACK=1`) |

### Why recurrent superblock helps under a byte cap

Independent 12-layer models pay **12×** for Q/K/V/O and MLP cores. Here, core linear maps are paid **6×**; only **cheap** per-virtual parameters (norms, gates, adapters) scale with 12. At equal artifact size, you can widen the network or spend more bytes on compression metadata and embeddings — the hypothesis is better **BPB per byte** than a flat 12L of the same budget.

## Data paths (important)

`train_gpt.py` resolves the **repository root** by searching upward from this file until it finds a `data/` directory (or use **`REPO_ROOT=/path/to/parameter-golf`**).

- If **`DATA_PATH`** or **`TOKENIZER_PATH`** point to a **missing** location (e.g. `/workspace/parameter-golf/...` while the clone lives at `/parameter-golf`), the script **warns** and falls back to `<repo>/data/...`.
- You still must run `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards …` once so `data/tokenizers/` and `data/datasets/` exist.

## Exact train command (8×H100 SXM)

From the **repository root** (after FineWeb shards + tokenizer are present under `data/`):

```bash
./records/track_10min_16mb/2026-03-24_jago_recurrent_superblock_record/run.sh
```

Or explicitly:

```bash
cd /path/to/parameter-golf
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-24_jago_recurrent_superblock_record/train_gpt.py
```

Recommended env (matches design flags — override as needed):

```bash
export SEED=1337
export RECURRENT_SUPERBLOCK=1 VIRTUAL_LAYERS=12 PHYSICAL_LAYERS=6
export DEPTH_ADAPTER_RANK=8
export VIRTUAL_DEPTH_DROPOUT=1 VDD_END_FULL_FRAC=0.25
export MODEL_DIM=576 MLP_MULT=3.5 NUM_HEADS=8 NUM_KV_HEADS=4
export XSA_LAST_N=12 PARTIAL_ROPE_DIM=16
export VR_ENABLED=1 GA_ENABLED=1 BIGRAM_HASH=2048
export EMA_ENABLED=1 EMA_DECAY=0.997
export QAT_START_FRAC=0.18 MIXED_INT5_INT6=1 PRUNE_FRAC=0.0
export USE_LZMA=1
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
export EVAL_STRIDE=64 SLIDING_EVAL_BATCH_SEQS=32
export ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500
export VAL_LOSS_EVERY=0          # set >0 if you want mid-train simple val
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
```

## Artifact accounting

Challenge total bytes = **UTF-8 bytes of this folder’s `train_gpt.py`** + **compressed weight payload**.

The script prints:

`ARTIFACT_BYTES total=... code=... compressed_weights=... (limit 16_000_000)`

It also writes `artifact_payload.pt` (compressed blob + meta) and `submission_metrics.json` (loss / BPB / byte estimate) beside this README.

**Target:** ~15.85–15.95MB before safety margin — tune `MODEL_DIM`, `BIGRAM_HASH`, `PRUNE_FRAC`, compressor, or int5 fraction if over cap.

## Ablation toggles (env)

| Flag | Effect |
|------|--------|
| `VIRTUAL_DEPTH_DROPOUT=0` | Always 12 virtual layers |
| `XSA_LAST_N=4` | XSA only on last 4 virtual layers |
| `BIGRAM_HASH=0` | Disable BigramHash |
| `MIXED_INT5_INT6=0` | All int6 rows (larger payload) |
| `USE_LZMA=0` | zlib level 9 instead of LZMA |
| `USE_ZSTD_FALLBACK=1` | zstd compression (needs `zstandard`) |
| `MODEL_DIM=640` | More capacity if artifact fits |
| `VE_ENABLED=1` + `VE_LAYERS=9,10` | Shared value embedding on listed **virtual** indices |

## Dependencies

- Repository `requirements.txt` at repo root (CUDA PyTorch, etc.).
- **Optional:** `flash_attn_interface` (FA3) on Hopper for best attention throughput; script falls back to `scaled_dot_product_attention`.
- **Optional:** `zstandard` if using `USE_ZSTD_FALLBACK=1`.

## Files in this folder

- `train_gpt.py` — training, eval, export (self-contained).
- `run.sh` — 8-GPU launcher from repo root.
- `submission.json` — leaderboard stub (fill after verified runs).
- `RESULTS.md` — 3-seed results template.
- `requirements-record.txt` — notes only (see above).
