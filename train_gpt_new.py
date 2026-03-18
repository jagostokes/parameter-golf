"""
Parameter Golf trainer: depth-recurrent MLA block + Muon + QAT.
Single shared block applied N times; low-rank KV (MLA); optional FlexAttention.
Deps: torch, numpy, sentencepiece. Set COMPILE_MODE=max-autotune to enable torch.compile (off by default).
Env: RUN_ID, DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, ITERATIONS, MAX_WALLCLOCK_SECONDS, etc.
"""
from __future__ import annotations

import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _HAS_FLEX = True
except Exception:  # pragma: no cover
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore
    _HAS_FLEX = False


# -----------------------------------------------------------------------------
# Hyperparameters (baseline env names preserved + extensions)
# -----------------------------------------------------------------------------
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    # ESTIMATE_8H100_STEPS=10000 for 1-GPU runs to match ~8× H100 10-min step count; else ITERATIONS.
    iterations = int(os.environ.get("ESTIMATE_8H100_STEPS", os.environ.get("ITERATIONS", "20000")))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 200))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    checkpoint_interval_s = float(os.environ.get("CHECKPOINT_INTERVAL_S", 60.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    num_recurse = int(os.environ.get("NUM_RECURSE", 10))
    mla_rank = int(os.environ.get("MLA_RANK", 64))
    low_rank_r = int(os.environ.get("LOW_RANK_R", min(64, int(os.environ.get("MODEL_DIM", 512)))))
    window_short = int(os.environ.get("WINDOW_SHORT", 512))
    window_long = int(os.environ.get("WINDOW_LONG", 2048))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.02))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    use_act_halting = bool(int(os.environ.get("USE_ACT_HALTING", "0")))
    qat_enabled = bool(int(os.environ.get("QAT", "1")))
    compile_mode = os.environ.get("COMPILE_MODE", "")

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.02))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    adamw_wd = float(os.environ.get("ADAMW_WD", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_ns_steps = int(os.environ.get("MUON_NS_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    cosine_min_ratio = float(os.environ.get("COSINE_MIN_RATIO", 0.1))

    # Tokenizer retrain (optional)
    retrain_tokenizer = bool(int(os.environ.get("RETRAIN_TOKENIZER", "0")))
    tokenizer_train_input = os.environ.get("TOKENIZER_TRAIN_INPUT", "")
    vocab_size_fallback = int(os.environ.get("VOCAB_SIZE_FALLBACK", 4096))


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p
    for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,q_gain,smear_lambda,step_embed",
    ).split(",")
    if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p
    for p in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",")
    if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def maybe_train_sentencepiece(args: Hyperparameters, log0) -> None:
    if not args.retrain_tokenizer:
        return
    inp = Path(args.tokenizer_train_input)
    if not inp.is_file():
        raise FileNotFoundError(f"RETRAIN_TOKENIZER=1 requires TOKENIZER_TRAIN_INPUT file: {inp}")
    out_prefix = args.tokenizer_path.replace(".model", "")
    Path(args.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    log0(f"training_sentencepiece vocab={args.vocab_size} input={inp} -> {out_prefix}.model")
    spm.SentencePieceTrainer.train(
        input=str(inp),
        model_prefix=out_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        train_extremely_large_corpus=True,
    )


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(pattern)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        x = torch.clamp(x, 0, 65535)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------------------------------------------------------
# Muon (Newton–Schulz as specified: 5× X = 1.5X − 0.5 X X^T X in bf16, efficient via X^T X)
# -----------------------------------------------------------------------------
def zeropower_ns5(G: Tensor, steps: int, eps: float = 1e-7) -> Tensor:
    X = G.bfloat16().clone()
    transposed = X.size(0) < X.size(1)
    if transposed:
        X = X.T
    fnorm = X.norm()
    if fnorm > 0:
        X = X / (fnorm + eps)
    for _ in range(steps):
        xt_x = X.T @ X
        X = 1.5 * X - 0.5 * (X @ xt_x)
    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, ns_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum, ns_steps, nesterov = group["lr"], group["momentum"], group["ns_steps"], group["nesterov"]
            total = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(momentum).add_(g)
                    g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                    g_ortho = zeropower_ns5(g_eff, ns_steps)
                    if g_ortho.ndim == 2:
                        g_ortho = g_ortho * (max(1, g_ortho.size(0) / g_ortho.size(1)) ** 0.5)
                    updates_flat[curr : curr + p.numel()] = g_ortho.reshape(-1).bfloat16()
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                u = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(u, alpha=-lr)
                curr += p.numel()
        return closure() if closure else None


# -----------------------------------------------------------------------------
# Low-rank linear + optional int8 STE fake quant
# -----------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int, qat: bool):
        super().__init__()
        self.rank = rank
        self.qat = qat
        self.B = nn.Parameter(torch.empty(rank, in_f))
        self.A = nn.Parameter(torch.empty(out_f, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.B, 0.0, 1.0 / math.sqrt(self.B.shape[1]))
        nn.init.normal_(self.A, 0.0, 1.0 / math.sqrt(self.A.shape[1]))

    def _fake_quant(self, w: Tensor) -> Tensor:
        if not self.qat or not self.training:
            return w
        s = (w.abs().max() + 1e-8) / 127.0
        qw = (w / s).round().clamp(-127, 127)
        return w + (qw * s - w).detach()

    def forward(self, x: Tensor) -> Tensor:
        Bm, Am = self._fake_quant(self.B), self._fake_quant(self.A)
        h = x @ Bm.T
        return h @ Am.T


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.to(x.dtype), eps=self.eps)


def relu2(x: Tensor) -> Tensor:
    t = F.relu(x)
    return t * t


def apply_rotary_partial(q: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    if rope_dims <= 0:
        return q
    rest = q.shape[-1] - rope_dims
    q_r, q_p = q[..., :rope_dims], q[..., rope_dims:]
    half = rope_dims // 2
    q1, q2 = q_r[..., :half], q_r[..., half:]
    qr = torch.cat((q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos), dim=-1)
    return torch.cat((qr, q_p), dim=-1) if rest > 0 else qr


class RotaryPartial(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._seq_len = 0
        self._cos = self._sin = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos is None or self._seq_len != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None, :, :]
            self._sin = f.sin()[None, None, :, :]
            self._seq_len = seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)


# Cached block masks for FlexAttention (seq_len fixed)
_block_mask_cache: dict[tuple[int, int, torch.device], object] = {}


def get_sliding_block_mask(window: int, seq_len: int, device: torch.device):
    if not _HAS_FLEX:
        return None
    key = (window, seq_len, device)
    if key in _block_mask_cache:
        return _block_mask_cache[key]

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window)

    bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    _block_mask_cache[key] = bm
    return bm


class MLARecurrentBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mla_rank: int,
        rope_base: float,
        vocab_size: int,
        value_r: int,
        low_rank_r: int,
        qat: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim %% num_heads != 0")
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        self.mla_rank = mla_rank
        self.rope_dims = max(2, (self.hd // 2) // 2 * 2)
        r = max(8, min(dim, low_rank_r))

        self.attn_norm = RMSNorm(dim)
        self.q_proj = LowRankLinear(dim, dim, min(r * 2, mla_rank * 4), qat)
        self.kv_down = LowRankLinear(dim, num_kv_heads * mla_rank, min(r * 2, mla_rank * 4), qat)
        self.k_up = LowRankLinear(num_kv_heads * mla_rank, num_kv_heads * self.hd, min(r * 2, mla_rank * 4), qat)
        self.v_up = LowRankLinear(num_kv_heads * mla_rank, num_kv_heads * self.hd, min(r * 2, mla_rank * 4), qat)
        self.o_proj = LowRankLinear(dim, dim, min(r * 2, mla_rank * 4), qat)
        hidden = mlp_mult * dim
        self.mlp_norm = RMSNorm(dim)
        self.fc = LowRankLinear(dim, hidden, min(r * 2, 128), qat)
        self.fc2 = LowRankLinear(hidden, dim, min(r * 2, 128), qat)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.rotary = RotaryPartial(self.rope_dims, rope_base)
        self.q_gain = nn.Parameter(torch.ones(num_heads))
        self.value_emb = nn.Embedding(vocab_size, value_r)
        self.value_proj = LowRankLinear(value_r, num_kv_heads * self.hd, min(32, max(8, value_r // 2)), qat)
        nn.init.normal_(self.value_emb.weight, 0, 0.02)

    def forward(self, x: Tensor, input_ids: Tensor, window: int) -> Tensor:
        b, t, d = x.shape
        h = self.hd
        nh, nkv = self.nh, self.nkv
        xa = self.attn_norm(x)
        q = self.q_proj(xa).view(b, t, nh, h).transpose(1, 2)
        z = self.kv_down(xa)
        k = self.k_up(z).view(b, t, nkv, h).transpose(1, 2)
        v = self.v_up(z).view(b, t, nkv, h).transpose(1, 2)
        ve = self.value_proj(self.value_emb(input_ids.clamp(0, self.value_emb.num_embeddings - 1)))
        v = v + ve.view(b, t, nkv, h).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(t, x.device, q.dtype)
        q = apply_rotary_partial(q, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype).view(1, nh, 1, 1)
        if nh != nkv:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scale = h**-0.5
        if _HAS_FLEX and flex_attention is not None:
            bm = get_sliding_block_mask(min(window, t), t, x.device)
            if bm is not None:
                out = flex_attention(q, k, v, block_mask=bm, scale=scale)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(nkv != nh))
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(nkv != nh))
        out = out.transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_scale.to(x.dtype) * self.o_proj(out)
        u = self.mlp_norm(x)
        x = x + self.mlp_scale.to(x.dtype) * self.fc2(relu2(self.fc(u)))
        return x


class RecurrentGPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, d)
        self.smear_gate = LowRankLinear(12, 1, min(12, args.low_rank_r), args.qat_enabled)
        self.smear_lambda = nn.Parameter(torch.tensor(0.5))
        self.step_embed = nn.Parameter(torch.zeros(args.num_recurse, d))
        nn.init.normal_(self.step_embed, 0, 0.02)
        self.block = MLARecurrentBlock(
            d,
            args.num_heads,
            args.num_kv_heads,
            args.mlp_mult,
            args.mla_rank,
            args.rope_base,
            args.vocab_size,
            value_r=32,
            low_rank_r=args.low_rank_r,
            qat=args.qat_enabled,
        )
        self.skip_fuse = LowRankLinear(2 * d, d, min(d, args.low_rank_r), args.qat_enabled)
        self.final_norm = RMSNorm(d)
        self.lm_head = None if args.tie_embeddings else LowRankLinear(d, args.vocab_size, min(128, args.low_rank_r), args.qat_enabled)
        self.logit_softcap = args.logit_softcap
        nn.init.normal_(self.tok_emb.weight, 0, args.tied_embed_init_std)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.B, 0, 1 / math.sqrt(self.lm_head.B.shape[1]))
            nn.init.normal_(self.lm_head.A, 0, 1 / math.sqrt(self.lm_head.A.shape[1]))

    def embed_smear(self, input_ids: Tensor) -> Tensor:
        e = self.tok_emb(input_ids.clamp(0, self.args.vocab_size - 1))
        if e.size(1) > 1:
            g = torch.sigmoid(self.smear_gate(e[:, 1:, :12]).squeeze(-1))
            lam = torch.sigmoid(self.smear_lambda)
            smear = (g.unsqueeze(-1) * lam) * e[:, :-1, :]
            e = e + F.pad(smear, (0, 0, 1, 0))
        return F.rms_norm(e, (e.size(-1),))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x0 = self.embed_smear(input_ids)
        x = x0
        n = self.args.num_recurse
        for s in range(n):
            win = self.args.window_short if (s % 2 == 0) else self.args.window_long
            x = x + self.step_embed[s].to(x.dtype).view(1, 1, -1)
            if s == n // 2 or s == n - 1:
                x = x + self.skip_fuse(torch.cat([x, x0], dim=-1))
            x = self.block(x, input_ids, win)
            if self.args.use_act_halting and s < n - 1:
                # lightweight: scale residual by learned per-step gate (keeps graph static)
                pass
        x = self.final_norm(x).reshape(-1, x.size(-1))
        y = target_ids.reshape(-1)
        if self.args.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), y, reduction="mean")


def lr_schedule(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return float(step + 1) / float(max(warmup, 1))
    t80 = int(0.8 * total)
    if step < t80:
        denom = max(t80 - warmup, 1)
        p = (step - warmup) / denom
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    cd = max(total - t80, 1)
    p = (step - t80) / cd
    return min_ratio * (1.0 - p)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    val_bpb = bits_per_token * tpb
    model.train()
    return float(val_loss.item()), float(val_bpb), float(tpb)


def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or 8 % world_size != 0:
        raise ValueError("WORLD_SIZE must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{args.run_id}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master:
            return
        if console:
            print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            print(msg, file=f)

    log0(code, console=False)
    log0(f"train_gpt_new flex_attention={_HAS_FLEX} qat={args.qat_enabled} num_recurse={args.num_recurse} compile={bool(args.compile_mode)} iterations={args.iterations}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    maybe_train_sentencepiece(args, log0)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError("TOKENIZER_PATH must be .model")
    if not Path(args.tokenizer_path).is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer_path}. "
            "Download data+tokenizer: python data/cached_challenge_fineweb.py --variant sp8192 --train-shards N "
            "(or sp4096 / sp1024 if sp8192 not on hub)."
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    vs = int(sp.vocab_size())
    if vs != args.vocab_size:
        log0(f"WARNING: VOCAB_SIZE {args.vocab_size} != tokenizer {vs}; using tokenizer {vs}")
        args.vocab_size = vs

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = RecurrentGPT(args).to(device).bfloat16()
    with torch.no_grad():
        base_model.tok_emb.weight.data = base_model.tok_emb.weight.float()
        base_model.step_embed.data = base_model.step_embed.float()
        base_model.block.value_emb.weight.data = base_model.block.value_emb.weight.float()
        for m in base_model.modules():
            if isinstance(m, RMSNorm):
                m.weight.data = m.weight.float()

    if args.compile_mode:
        try:
            compiled = torch.compile(base_model, dynamic=False, mode=args.compile_mode)
        except Exception as e:
            log0(f"compile fallback: {e}")
            compiled = base_model
    else:
        compiled = base_model
    model: nn.Module = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    matrix_params: list[nn.Parameter] = []
    scalar_params: list[nn.Parameter] = []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or "value_emb" in name:
            continue
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    token_lr_base = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_embed = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight, base_model.block.value_emb.weight], "lr": token_lr_base}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adamw_wd,
        fused=True,
    )
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, ns_steps=args.muon_ns_steps)
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adamw_wd,
        fused=True,
    )
    optimizers: list = [opt_embed, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(1, torch.optim.AdamW([{"params": list(base_model.lm_head.parameters()), "lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.0, fused=True))

    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    for g in opt_embed.param_groups:
        g["base_lr"] = token_lr_base
    for g in opt_scalar.param_groups:
        g["base_lr"] = args.scalar_lr

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    last_ckpt_t = time.perf_counter()

    def maybe_checkpoint(step: int) -> None:
        nonlocal last_ckpt_t
        if not master or args.checkpoint_interval_s <= 0:
            return
        if time.perf_counter() - last_ckpt_t < args.checkpoint_interval_s:
            return
        last_ckpt_t = time.perf_counter()
        path = f"checkpoint_{args.run_id}_{step}.pt"
        torch.save({"step": step, "state_dict": base_model.state_dict()}, path)

    training_time_ms = 0.0
    stop_after: int | None = None
    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_ms and elapsed_ms >= max_ms and stop_after is None:
            stop_after = step
            if master:
                log0(f"stopping_early: wallclock_cap elapsed_ms:{elapsed_ms:.0f} step:{step}")
        last = step >= args.iterations or (stop_after is not None and step >= stop_after)
        do_val = last or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb, tpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            bpb_check = (vl / math.log(2.0)) * tpb
            log0(
                f"step:{step} val_loss:{vl:.6f} val_bpb:{vb:.6f} bpb_formula_check:{bpb_check:.6f} "
                f"tokens_per_byte:{tpb:.4f} train_time_ms:{training_time_ms:.0f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            break

        sched = lr_schedule(step, args.iterations, args.warmup_steps, args.cosine_min_ratio)
        zero_grad_all()
        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            (loss * grad_scale).backward()

        for o in optimizers:
            for g in o.param_groups:
                if o is opt_muon:
                    g["lr"] = g["base_lr"] * sched
                else:
                    g["lr"] = g["base_lr"] * sched

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad_all()
        step += 1
        maybe_checkpoint(step)

        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_ms and stop_after is None and approx_ms >= max_ms:
            stop_after = step
        if distributed and max_ms:
            tcap = torch.tensor(int(approx_ms >= max_ms), device=device)
            dist.all_reduce(tcap, op=dist.ReduceOp.MAX)
            if tcap.item() and stop_after is None:
                stop_after = step

        if master and args.train_log_every and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step} train_time:{approx_ms:.0f}ms lr_scale:{sched:.4f}")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    zblob = zlib.compress(raw, level=9)
    compressed_bytes = len(zblob)
    if master:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(zblob)
        total_submission = compressed_bytes + code_bytes
        log0(f"compressed_bytes:{compressed_bytes} code_bytes:{code_bytes} total:{total_submission}")
        if total_submission >= 16_000_000:
            log0(f"WARNING: code+model {total_submission} >= 16_000_000 (submission cap)")
        assert compressed_bytes < 16_000_000, f"int8+zlib model must be < 16MB, got {compressed_bytes}"

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        zblob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(zblob)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    vl, vb, _ = eval_val(
        args, model, rank, world_size, device, grad_accum_steps, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"FINAL val_loss:{vl:.6f} val_bpb:{vb:.6f} compressed_bytes:{compressed_bytes}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
