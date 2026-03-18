# Parameter Golf autoresearch program

**TASK:** Minimize val_bpb on FineWeb val set, 16MB artifact limit, 10min on 8xH100

**METRIC:** bits per byte = (cross_entropy / ln2) * (tokens / raw_bytes)

**BASELINE:** 9-layer, 512-dim, 1024-vocab, tied embeddings, 4 KV heads → **1.2244 bpb** (200-step proxy baseline: run `train_gpt.py` locally)

## KNOWN WINS (update as you discover them)

- _(empty — current codebase already stacks Muon, RoPE, ReLU², QK-norm, U-Net skips vs naive baseline)_

## KNOWN LOSSES (update as you discover them)

- _(empty at start)_

## CURRENT BEST BPB

**1.2244**

## CURRENT BEST CONFIG

`train_gpt.py` defaults: `fineweb10B_sp1024`, vocab 1024, 9×512, 4 KV heads, tied embeddings, Muon+Adam split.

---

**Notes:** Published HF `willdepueoai/parameter-golf` ships **only** `fineweb10B_sp1024`. SP2048 requires local export via `data/download_hf_docs_and_tokenize.py` + `data/tokenizer_specs_sp2048.json` (optional `--proxy-max-docs` for fast proxy runs).
