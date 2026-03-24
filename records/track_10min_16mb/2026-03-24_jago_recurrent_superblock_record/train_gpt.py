# Recurrent superblock record attempt — self-contained (no TTT). See README.md in this folder.
from __future__ import annotations

import copy
import glob
import io
import json
import lzma
import math
import os
import pickle
import random
import struct
import subprocess
import sys
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

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _flash_attn_3_func = None
    _HAS_FA3 = False

try:
    import zstandard
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

# -----------------------------------------------------------------------------
# Env / hyperparameters
# -----------------------------------------------------------------------------

def _b(name: str, default: str) -> bool:
    return bool(int(os.environ.get(name, default)))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = _i("SEED", 1337)

    val_batch_size = _i("VAL_BATCH_SIZE", 524_288)
    val_loss_every = _i("VAL_LOSS_EVERY", 4000)
    train_log_every = _i("TRAIN_LOG_EVERY", 500)
    iterations = _i("ITERATIONS", 20000)
    warmdown_iters = _i("WARMDOWN_ITERS", 3500)
    warmup_steps = _i("WARMUP_STEPS", 20)
    train_batch_tokens = _i("TRAIN_BATCH_TOKENS", 786_432)
    train_seq_len = _i("TRAIN_SEQ_LEN", 2048)
    eval_seq_len = _i("EVAL_SEQ_LEN", 2048)
    max_wallclock_seconds = _f("MAX_WALLCLOCK_SECONDS", 600.0)

    recurrent_superblock = _b("RECURRENT_SUPERBLOCK", "1")
    num_virtual = _i("VIRTUAL_LAYERS", 12)
    num_physical = _i("PHYSICAL_LAYERS", 6)
    depth_adapter_rank = _i("DEPTH_ADAPTER_RANK", 8)
    virtual_depth_dropout = _b("VIRTUAL_DEPTH_DROPOUT", "1")
    vdd_start_keep = _f("VDD_START_KEEP", 0.67)  # unused; we use randint 8–10
    vdd_end_full_frac = _f("VDD_END_FULL_FRAC", 0.25)

    model_dim = _i("MODEL_DIM", 576)
    num_heads = _i("NUM_HEADS", 8)
    num_kv_heads = _i("NUM_KV_HEADS", 4)
    mlp_mult = _f("MLP_MULT", 3.5)
    vocab_size = _i("VOCAB_SIZE", 1024)
    tie_embeddings = _b("TIE_EMBEDDINGS", "1")
    rope_base = _f("ROPE_BASE", 10000.0)
    logit_softcap = _f("LOGIT_SOFTCAP", 30.0)
    qk_gain_init = _f("QK_GAIN_INIT", 1.5)

    xsa_last_n = _i("XSA_LAST_N", 12)
    partial_rope_dim = _i("PARTIAL_ROPE_DIM", 16)
    ln_scale = _b("LN_SCALE", "1")
    bigram_vocab_size = _i("BIGRAM_HASH", _i("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = _i("BIGRAM_DIM", 128)
    vr_enabled = _b("VR_ENABLED", "1")
    ga_enabled = _b("GA_ENABLED", "1")

    ema_enabled = _b("EMA_ENABLED", "1")
    ema_decay = _f("EMA_DECAY", 0.997)
    qat_start_frac = _f("QAT_START_FRAC", 0.18)
    mixed_int5_int6 = _b("MIXED_INT5_INT6", "1")
    prune_frac = _f("PRUNE_FRAC", 0.0)
    use_lzma = _b("USE_LZMA", "1")
    use_zstd_fallback = _b("USE_ZSTD_FALLBACK", "0")

    eval_stride = _i("EVAL_STRIDE", 64)
    sliding_eval_batch_seqs = _i("SLIDING_EVAL_BATCH_SEQS", 32)

    matrix_lr = _f("MATRIX_LR", 0.025)
    scalar_lr = _f("SCALAR_LR", 0.025)
    tied_embed_lr = _f("TIED_EMBED_LR", 0.035)
    tied_embed_init_std = _f("TIED_EMBED_INIT_STD", 0.005)
    muon_momentum = _f("MUON_MOMENTUM", 0.99)
    muon_backend_steps = _i("MUON_BACKEND_STEPS", 5)
    muon_momentum_warmup_start = _f("MUON_MOMENTUM_WARMUP_START", 0.92)
    muon_momentum_warmup_steps = _i("MUON_MOMENTUM_WARMUP_STEPS", 1500)
    beta1 = _f("BETA1", 0.9)
    beta2 = _f("BETA2", 0.95)
    adam_eps = _f("ADAM_EPS", 1e-8)
    grad_clip_norm = _f("GRAD_CLIP_NORM", 0.3)
    muon_wd = _f("MUON_WD", 0.04)
    adam_wd = _f("ADAM_WD", 0.04)

    ve_enabled = _b("VE_ENABLED", "0")
    ve_dim = _i("VE_DIM", 128)
    ve_layers = os.environ.get("VE_LAYERS", "9,10")


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "ln_w",
    "adapt_",
    "attn_gate",
    "vr_lambda",
    "gate",
    "scale",
    "ve_layer",
)


# -----------------------------------------------------------------------------
# Newton–Schulz + Muon (batched 3D bank tensors)
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class MuonBanks(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        # Gradients already averaged across ranks in the training loop.
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.bfloat16()
                if wd > 0:
                    g = g + wd * p.bfloat16()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                gupd = buf.mul(momentum).add_(g) if nesterov else buf
                if p.ndim == 3:
                    M, N = gupd.shape[-2], gupd.shape[-1]
                    sc = max(1.0, float(M) / float(N)) ** 0.5
                    upd = zeropower_via_newtonschulz5(gupd.float(), steps=group["backend_steps"]) * sc
                else:
                    upd = zeropower_via_newtonschulz5(gupd.float(), steps=group["backend_steps"])
                    if p.ndim == 2:
                        upd = upd * max(1.0, float(p.shape[0]) / float(p.shape[1])) ** 0.5
                p.add_(upd.to(dtype=p.dtype), alpha=-lr)
        return None


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        out: list[Tensor] = []
        rem = n
        while rem > 0:
            if self.pos >= self.tokens.numel():
                self._advance()
                continue
            k = min(rem, self.tokens.numel() - self.pos)
            out.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
        return out[0] if len(out) == 1 else torch.cat(out)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = chunk[start : start + span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(pattern)
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_np, dtype=torch.bool, device=device),
    )


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------

class RMSNormW(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, ln_scale_factor: float = 1.0) -> Tensor:
        y = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return y * (self.weight * ln_scale_factor).to(dtype=y.dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((xr, xp), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float, train_seq_len: int, rope_dims: int):
        super().__init__()
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.base = base
        self._cos = self._sin = None
        self._slen = 0

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._slen != seq_len or self._cos is None or self._cos.device != device:
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                rd = self.rope_dims
                nb = self.base * (scale ** (rd / max(rd - 2, 1)))
                inv = 1.0 / (nb ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv.dtype)
            freqs = torch.outer(t, inv)
            self._cos = freqs.cos()[None, :, None, :]
            self._sin = freqs.sin()[None, :, None, :]
            self._slen = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)


class DepthAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        xp = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * xp


class BigramHashEmbedding(nn.Module):
    def __init__(self, n_buckets: int, edim: int, model_dim: int):
        super().__init__()
        self.n_buckets = n_buckets
        self.emb = nn.Embedding(n_buckets, edim)
        nn.init.zeros_(self.emb.weight)
        self.proj = nn.Linear(edim, model_dim, bias=False) if edim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    @staticmethod
    def hash_ids(toks: Tensor, mod: int) -> Tensor:
        t = toks.to(torch.int32)
        m = mod - 1
        o = torch.empty_like(t)
        o[..., 0] = m
        o[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % m
        return o.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.emb(self.hash_ids(token_ids, self.n_buckets))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab: int, edim: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, edim)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.proj = nn.Linear(edim, out_dim, bias=False) if edim != out_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.emb(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


def _fake_quant_weight(w: Tensor, bits: int) -> Tensor:
    w32 = w.float()
    clip = 2 ** (bits - 1) - 1
    row_max = w32.abs().amax(dim=1).clamp_min(1e-8)
    scale = (row_max / clip).clamp_min(1e-8)
    q = torch.round(w32 / scale[:, None]).clamp(-clip, clip)
    recon = q * scale[:, None]
    return w + (recon - w).detach()


_QAT_ACTIVE = False
_QAT_BITS = 6


class RecurrentGPT(nn.Module):
    """6 physical blocks × 2 unrolls = 12 virtual layers; shared Q/K/V/O/MLP per physical index."""

    def __init__(self, args: Hyperparameters):
        super().__init__()
        d = args.model_dim
        nh = args.num_heads
        nkv = args.num_kv_heads
        if d % nh:
            raise ValueError("model_dim %% num_heads != 0")
        self.head_dim = d // nh
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        self.kv_dim = nkv * self.head_dim
        mlp_dim = int(round(args.mlp_mult * d))
        self.mlp_dim = mlp_dim
        self.num_virtual = args.num_virtual
        self.num_physical = args.num_physical
        if self.num_virtual != 2 * self.num_physical:
            raise ValueError("Expect VIRTUAL_LAYERS == 2 * PHYSICAL_LAYERS")
        self.num_encoder = self.num_virtual // 2
        self.num_decoder = self.num_virtual - self.num_encoder
        self.n_skip = min(self.num_encoder, self.num_decoder)
        self.rope_dims = args.partial_rope_dim if args.partial_rope_dim > 0 else self.head_dim
        self.ln_scale_global = args.ln_scale
        self.value_residual = args.vr_enabled
        self.gated_attention = args.ga_enabled

        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = args.logit_softcap
        self.tied_embed_init_std = args.tied_embed_init_std

        self.tok_emb = nn.Embedding(args.vocab_size, d)
        self.bigram = (
            BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, d)
            if args.bigram_vocab_size > 0
            else None
        )
        self.smear = SmearGate(d)

        # Banks: Q[0:6], O[6:12], K[0:6], V[6:12], MLP up/down per physical
        self.qo_bank = nn.Parameter(torch.empty(12, d, d))
        self.kv_bank = nn.Parameter(torch.empty(12, self.kv_dim, d))
        self.mlp_up_bank = nn.Parameter(torch.empty(self.num_physical, mlp_dim, d))
        self.mlp_down_bank = nn.Parameter(torch.empty(self.num_physical, d, mlp_dim))

        self.q_gains = nn.Parameter(torch.full((self.num_physical, nh), args.qk_gain_init, dtype=torch.float32))
        self.skip_weights = nn.Parameter(torch.ones(self.n_skip, d, dtype=torch.float32))

        self.rotary = Rotary(self.head_dim, args.rope_base, args.train_seq_len, self.rope_dims)

        # Per-virtual blocks: norms, scales, resid, adapter
        self.attn_norms = nn.ModuleList([RMSNormW(d) for _ in range(self.num_virtual)])
        self.mlp_norms = nn.ModuleList([RMSNormW(d) for _ in range(self.num_virtual)])
        self.attn_scales = nn.ParameterList([nn.Parameter(torch.ones(d, dtype=torch.float32)) for _ in range(self.num_virtual)])
        self.mlp_scales = nn.ParameterList([nn.Parameter(torch.ones(d, dtype=torch.float32)) for _ in range(self.num_virtual)])
        self.resid_mixes = nn.ParameterList(
            [nn.Parameter(torch.stack((torch.ones(d), torch.zeros(d))).float()) for _ in range(self.num_virtual)]
        )
        self.adapters = nn.ModuleList([DepthAdapter(d, args.depth_adapter_rank) for _ in range(self.num_virtual)])

        self.use_xsa = [False] * self.num_virtual
        if args.xsa_last_n > 0:
            for i in range(max(0, self.num_virtual - args.xsa_last_n), self.num_virtual):
                self.use_xsa[i] = True

        self.ve_layer_indices = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []
        kv_ve = self.kv_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(args.vocab_size, args.ve_dim, kv_ve)
            self.ve_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_scales = nn.ParameterList()

        if self.gated_attention:
            self.attn_gates = nn.ModuleList([nn.Linear(d, nh, bias=True) for _ in range(self.num_virtual)])
            for m in self.attn_gates:
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, 4.0)
        else:
            self.attn_gates = None

        if self.value_residual:
            self.vr_lambdas = nn.ParameterList(
                [nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32)) for _ in range(self.num_virtual)]
            )
        else:
            self.vr_lambdas = None

        self.final_norm = RMSNormW(d)
        self.lm_head = None if args.tie_embeddings else nn.Linear(d, args.vocab_size, bias=False)
        if self.lm_head is not None:
            nn.init.zeros_(self.lm_head.weight)

        self._init_weights()
        self._register_qat_hooks()

    def _register_qat_hooks(self):
        self._bank_names = ("qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank")

    def _maybe_fq(self, w: Tensor) -> Tensor:
        if not _QAT_ACTIVE or not self.training:
            return w
        return _fake_quant_weight(w, _QAT_BITS)

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=self.tied_embed_init_std)
        nph = self.num_physical
        ps = 1.0 / math.sqrt(2 * self.num_virtual)
        for p in range(nph):
            nn.init.orthogonal_(self.qo_bank.data[p], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[6 + p])
            nn.init.orthogonal_(self.kv_bank.data[p], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[6 + p], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[p], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[p])
            self.qo_bank.data[6 + p].mul_(ps)
            self.mlp_down_bank.data[p].mul_(ps)

    def _ve(self, v: int, input_ids: Tensor, cache: dict) -> Tensor | None:
        if self.ve_shared is None or v not in self.ve_layer_indices:
            return None
        if "ve" not in cache:
            cache["ve"] = self.ve_shared(input_ids)
        idx = self.ve_layer_indices.index(v)
        return cache["ve"] * self.ve_scales[idx].to(dtype=cache["ve"].dtype)

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        g = H // Hkv
        yg = y.reshape(B, T, Hkv, g, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (yg * vn).sum(dim=-1, keepdim=True) * vn
        return (yg - proj).reshape(B, T, H, D)

    def _attn(self, x: Tensor, vidx: int, p: int, v_embed: Tensor | None, v0: Tensor | None):
        d = self.head_dim
        nh, nkv = self.q_gains.size(1), self.kv_dim // d
        bsz, tlen, dim = x.shape
        qw = self._maybe_fq(self.qo_bank[p])
        ow = self._maybe_fq(self.qo_bank[6 + p])
        kw = self._maybe_fq(self.kv_bank[p])
        vw = self._maybe_fq(self.kv_bank[6 + p])
        q = F.linear(x, qw.to(x.dtype)).reshape(bsz, tlen, nh, d)
        k = F.linear(x, kw.to(x.dtype)).reshape(bsz, tlen, nkv, d)
        vv = F.linear(x, vw.to(x.dtype))
        if v_embed is not None:
            vv = vv + v_embed
        vv = vv.reshape(bsz, tlen, nkv, d)
        raw_v = vv if self.value_residual else None
        if self.value_residual and v0 is not None and self.vr_lambdas is not None:
            lam = self.vr_lambdas[vidx].to(dtype=vv.dtype)
            vv = lam[0] * v0 + lam[1] * vv
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(tlen, x.device, q.dtype)
        q = apply_rope(q, cos, sin, self.rope_dims)
        k = apply_rope(k, cos, sin, self.rope_dims)
        q = q * self.q_gains[p].to(q.dtype)[None, None, :, None]

        if _HAS_FA3:
            y = _flash_attn_3_func(q, k, vv, causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2),
                is_causal=True, enable_gqa=(nkv != nh),
            ).transpose(1, 2)
        if self.use_xsa[vidx]:
            y = self._xsa(y, vv)
        if self.gated_attention and self.attn_gates is not None:
            g = torch.sigmoid(self.attn_gates[vidx](x)).unsqueeze(-1)
            y = y * g
        y = y.reshape(bsz, tlen, dim)
        return F.linear(y, ow.to(x.dtype)), raw_v

    def _mlp(self, x: Tensor, p: int) -> Tensor:
        up = self._maybe_fq(self.mlp_up_bank[p])
        dn = self._maybe_fq(self.mlp_down_bank[p])
        h = F.leaky_relu(F.linear(x, up.to(x.dtype)), negative_slope=0.5)
        return F.linear(h.square(), dn.to(x.dtype))

    def _block(self, vidx: int, x: Tensor, x0: Tensor, v0: Tensor | None, ve: Tensor | None):
        p = vidx // 2
        lsf = 1.0 / math.sqrt(vidx + 1) if self.ln_scale_global else 1.0
        mix = self.resid_mixes[vidx].to(dtype=x.dtype)
        xh = mix[0] * x + mix[1] * x0
        an = self.attn_norms[vidx](xh, lsf)
        ao, raw_v = self._attn(an, vidx, p, ve, v0)
        xh = xh + self.attn_scales[vidx].to(x.dtype) * ao
        xh = xh + self.adapters[vidx](xh)
        mn = self.mlp_norms[vidx](xh, lsf)
        xh = xh + self.mlp_scales[vidx].to(x.dtype) * self._mlp(mn, p)
        return xh, raw_v

    def forward_body(self, input_ids: Tensor, vdd_n_keep: int | None) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        keep = self.num_virtual if vdd_n_keep is None else max(1, min(vdd_n_keep, self.num_virtual))
        ve_cache: dict = {}

        for v in range(min(keep, self.num_encoder)):
            ve = self._ve(v, input_ids, ve_cache)
            x, rv = self._block(v, x, x0, v0, ve)
            if v0 is None and rv is not None:
                v0 = rv
            skips.append(x)

        dec_count = max(0, keep - self.num_encoder)
        for di in range(dec_count):
            v = self.num_encoder + di
            if self.n_skip > di and skips:
                x = x + self.skip_weights[di].to(x.dtype) * skips.pop()
            ve = self._ve(v, input_ids, ve_cache)
            x, rv = self._block(v, x, x0, v0, ve)
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor, vdd_n_keep: int | None = None) -> Tensor:
        x = self.forward_body(input_ids, vdd_n_keep)
        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor, vdd_n_keep: int | None = None) -> Tensor:
        x = self.forward_body(input_ids, vdd_n_keep)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------

def eval_val_simple(
    args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device,
    grad_accum_steps: int, val_tokens: Tensor, base_b: Tensor, hls: Tensor, bnd: Tensor,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    local_bt = args.val_batch_size // (world_size * grad_accum_steps)
    local_seqs = local_bt // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be = min(bs + local_seqs, s1)
            raw = val_tokens[bs * seq_len : be * seq_len + 1].to(device, dtype=torch.int64)
            x = raw[:-1].reshape(-1, seq_len)
            y = raw[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, None)
            n = float(y.numel())
            loss_sum = loss_sum + loss.double() * n
            tok_cnt += n
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_b[tgt].double()
            tb = tb + (hls[tgt] & ~bnd[prev]).double()
            byte_cnt += tb.sum()
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_cnt).item()
    bpb = vl / math.log(2.0) * (tok_cnt.item() / byte_cnt.item())
    model.train()
    return float(vl), float(bpb)


def eval_sliding(
    args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device,
    val_tokens: Tensor, base_b: Tensor, hls: Tensor, bnd: Tensor, stride: int,
) -> tuple[float, float]:
    seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    total_tok = val_tokens.numel() - 1
    starts = [ws for ws in range(0, total_tok, stride) if min(ws + seq_len, total_tok) - ws >= 1]
    nw = len(starts)
    ms = (nw * rank) // world_size
    me = (nw * (rank + 1)) // world_size
    my = starts[ms:me]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    bs_max = args.sliding_eval_batch_seqs
    with torch.inference_mode():
        for bi in range(0, len(my), bs_max):
            batch_ws = my[bi : bi + bs_max]
            bsz = len(batch_ws)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tok)
                wlen = end - ws
                wlens.append(wlen)
                ch = val_tokens[ws : end + 1].to(device, dtype=torch.int64)
                xb[i, :wlen] = ch[:-1]
                yb[i, :wlen] = ch[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(xb, None)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum += nll[i, s:wlen].double().sum()
                tok_cnt += float(wlen - s)
                tgt, prev = yb[i, s:wlen], xb[i, s:wlen]
                tb = base_b[tgt].double()
                tb = tb + (hls[tgt] & ~bnd[prev]).double()
                byte_cnt += tb.sum()
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_cnt).item()
    bpb = vl / math.log(2.0) * (tok_cnt.item() / byte_cnt.item())
    model.train()
    return float(vl), float(bpb)


# -----------------------------------------------------------------------------
# Quantization + export
# -----------------------------------------------------------------------------

def quantize_rowwise_mixed(
    t: Tensor, row_bits: Tensor, clip6: int = 31, clip5: int = 15,
) -> tuple[Tensor, Tensor, Tensor]:
    """row_bits: int8 per row, 5 or 6."""
    t32 = t.float()
    qrows = []
    scales = []
    for r in range(t32.shape[0]):
        b = int(row_bits[r].item())
        clip = clip6 if b >= 6 else clip5
        row = t32[r]
        mx = row.abs().max().clamp_min(1e-8)
        sc = (mx / clip).clamp_min(1e-8)
        qi = torch.round(row / sc).clamp(-clip, clip).to(torch.int8)
        qrows.append(qi)
        scales.append(sc.to(torch.float16))
    return torch.stack(qrows), torch.stack(scales), row_bits.to(torch.int8)


def sensitivity_rows(w: Tensor) -> Tensor:
    """Cheap proxy: mean abs * std per row."""
    wf = w.float()
    return wf.abs().mean(dim=1) * wf.std(dim=1).clamp_min(1e-8)


def export_state_dict(
    sd: dict[str, Tensor],
    args: Hyperparameters,
    code_bytes: bytes,
) -> tuple[bytes, int, dict]:
    """Returns compressed payload, uncompressed tensor bytes, meta."""
    flat: dict[str, object] = {"__format__": "rsb_mixed_v1", "tensors": {}}
    raw_sz = 0
    meta_tensors: dict[str, dict] = {}

    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.ndim != 2 or t.numel() <= 4096:
            arr = t.float().numpy() if t.dtype in (torch.bfloat16, torch.float16) else t.numpy()
            flat["tensors"][name] = {"kind": "raw", "dtype": str(t.dtype), "data": arr}
            raw_sz += arr.nbytes
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "tok_emb" in name or "lm_head" in name \
                or "bigram.emb.weight" in name or "ve_shared.emb.weight" in name:
            arr = t.to(torch.float16).contiguous().numpy()
            flat["tensors"][name] = {"kind": "raw", "dtype": str(t.dtype), "data": arr}
            raw_sz += arr.nbytes
            continue

        sens = sensitivity_rows(t)
        median = sens.median()
        if args.mixed_int5_int6:
            bits = torch.where(sens >= median, torch.tensor(6), torch.tensor(5)).to(torch.int8)
        else:
            bits = torch.full((t.shape[0],), 6, dtype=torch.int8)

        if args.prune_frac > 0:
            thr = torch.quantile(t.abs().flatten(), args.prune_frac)
            t = t * (t.abs() >= thr).to(t.dtype)

        q, sc, rb = quantize_rowwise_mixed(t, bits)
        meta_tensors[name] = {"bits": rb.numpy().tobytes(), "shape": list(t.shape)}
        flat["tensors"][name] = {"kind": "q", "q": q.numpy(), "scale": sc.numpy(), "bits": rb.numpy().tobytes()}
        raw_sz += q.numel() + sc.numel() * 2 + rb.numel()

    bio = io.BytesIO()
    pickle.dump(flat, bio, protocol=4)
    raw_payload = bio.getvalue()
    if args.use_zstd_fallback and _HAS_ZSTD:
        cctx = zstandard.ZstdCompressor(level=19)
        comp = cctx.compress(raw_payload)
    elif args.use_lzma:
        comp = lzma.compress(raw_payload, preset=6)
    else:
        comp = zlib.compress(raw_payload, level=9)
    total = len(code_bytes) + len(comp)
    return comp, total, {"meta": meta_tensors, "payload_raw": len(raw_payload), "payload_comp": len(comp)}


def print_artifact_accounting(code_path: Path, comp: bytes, extra: dict):
    code_b = code_path.read_bytes()
    total = len(code_b) + len(comp)
    print(f"ARTIFACT_BYTES total={total} code={len(code_b)} compressed_weights={len(comp)} "
          f"raw_pickle={extra.get('payload_raw', 0)} (limit 16_000_000)")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def restore_scalars_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim < 2 or any(x in name for x in CONTROL_TENSOR_NAME_PATTERNS):
                if p.dtype != torch.float32:
                    p.data = p.data.float()


def sample_vdd_n_keep(args: Hyperparameters, step: int, total_steps: int) -> int:
    if not args.virtual_depth_dropout or total_steps <= 0:
        return args.num_virtual
    frac = step / max(total_steps, 1)
    if frac >= (1.0 - args.vdd_end_full_frac):
        return args.num_virtual
    lo = max(8, int(args.vdd_start_keep * args.num_virtual + 1e-6))
    hi = min(10, args.num_virtual)
    if lo > hi:
        lo = hi
    return random.randint(lo, hi)


def main() -> None:
    code_path = Path(__file__).resolve()
    code_bytes = code_path.read_text(encoding="utf-8").encode("utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError("WORLD_SIZE must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    def log0(msg: str, c: bool = True):
        if master:
            print(msg) if c else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE mismatch")
    val_sl = max(args.train_seq_len, args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_sl)
    base_b, hls, bnd = build_sentencepiece_luts(sp, args.vocab_size, device)

    global _QAT_ACTIVE, _QAT_BITS
    _QAT_ACTIVE = False

    model = RecurrentGPT(args).to(device=device, dtype=torch.bfloat16)
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    restore_scalars_fp32(model)

    matrix_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
    scalars = [p for n, p in model.named_parameters() if p.ndim < 2 or any(x in n for x in CONTROL_TENSOR_NAME_PATTERNS)]
    for n, p in model.named_parameters():
        if "skip_weights" in n and p not in scalars:
            scalars.append(p)
    scalars = list(dict.fromkeys(scalars))
    tok_params = [{"params": [model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}]
    if model.bigram is not None:
        tok_params.append({"params": [model.bigram.emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr})
        if model.bigram.proj is not None:
            scalars.append(model.bigram.proj.weight)
        scalars.append(model.bigram.scale)
    if model.ve_shared is not None:
        tok_params.append({"params": [model.ve_shared.emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr})
        if model.ve_shared.proj is not None:
            scalars.append(model.ve_shared.proj.weight)
        scalars.append(model.ve_shared.scale)
        for s in model.ve_scales:
            scalars.append(s)
    scalars.append(model.smear.gate)

    opt_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_muon = MuonBanks(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    opt_sc = torch.optim.AdamW(
        [{"params": scalars, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    opts = [opt_tok, opt_muon, opt_sc]
    if model.lm_head is not None:
        opt_h = torch.optim.Adam(
            [{"params": [model.lm_head.weight], "lr": 0.008, "base_lr": 0.008}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        opts.append(opt_h)
        scalars.append(model.lm_head.weight)

    ema_sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()} if args.ema_enabled else None
    ema_decay = args.ema_decay

    loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    train_ms = 0.0
    step = 0
    total_steps_est = args.iterations

    def lr_scale(s: int, elapsed: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - s) / max(args.warmdown_iters, 1), 0.0) if ws <= s < args.iterations else 1.0
        sp_ms = elapsed / max(s, 1)
        wd_ms = args.warmdown_iters * sp_ms
        rem = max(max_ms - elapsed, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if master:
        npar = sum(p.numel() for p in model.parameters())
        log0(f"recurrent_superblock virtual={args.num_virtual} physical={args.num_physical} params={npar} FA3={_HAS_FA3}")

    while True:
        last = step >= args.iterations
        if last:
            break
        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        sc = lr_scale(step, elapsed)
        qat_start_step = int(args.qat_start_frac * max(args.iterations, 1))
        if args.qat_start_frac > 0 and step >= qat_start_step:
            _QAT_ACTIVE = True

        n_keep = sample_vdd_n_keep(args, step, total_steps_est)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum_steps):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, n_keep)
            (loss * grad_scale).backward()
        if distributed:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        mm = args.muon_momentum
        if step < args.muon_momentum_warmup_steps:
            t = step / max(args.muon_momentum_warmup_steps, 1)
            mm = args.muon_momentum_warmup_start + t * (args.muon_momentum - args.muon_momentum_warmup_start)
        for g in opt_muon.param_groups:
            g["momentum"] = mm
        for opt in opts:
            for g in opt.param_groups:
                base = g.get("base_lr", g["lr"])
                g["lr"] = base * sc
            opt.step()

        if ema_sd is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_sd[k].mul_(ema_decay).add_(v.float(), alpha=1.0 - ema_decay)

        step += 1
        train_ms += 1000.0 * (time.perf_counter() - t0)
        t0 = time.perf_counter()

        if master and (step % args.train_log_every == 0 or step == 1):
            log0(f"step {step} loss~ train_ms={train_ms:.0f}ms vdd_keep={n_keep} qat={_QAT_ACTIVE}")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            if ema_sd is not None:
                bak = copy.deepcopy(model.state_dict())
                model.load_state_dict({k: v.to(device=device, dtype=torch.bfloat16) for k, v in ema_sd.items()}, strict=False)
            vl, vb = eval_val_simple(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_b, hls, bnd)
            if ema_sd is not None:
                model.load_state_dict(bak, strict=True)
            if master:
                log0(f"val step={step} val_loss={vl:.4f} val_bpb(simple)={vb:.4f}")

        if max_ms is not None and train_ms >= max_ms:
            if master:
                log0(f"wallclock stop step={step} train_ms={train_ms:.0f}")
            break

    # Final: EMA weights, sliding eval, export
    if ema_sd is not None:
        model.load_state_dict({k: v.to(device=device, dtype=torch.bfloat16) for k, v in ema_sd.items()}, strict=False)
    restore_scalars_fp32(model)

    vl, vb_sl = eval_sliding(args, model, rank, world_size, device, val_tokens, base_b, hls, bnd, args.eval_stride)
    if master:
        log0(f"final_sliding val_loss={vl:.6f} val_bpb={vb_sl:.6f} stride={args.eval_stride}")

    if master:
        sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        comp, total_b, ex = export_state_dict(sd_cpu, args, code_bytes)
        print_artifact_accounting(code_path, comp, ex)
        out_dir = code_path.parent
        torch.save({"compressed": comp, "extra": ex}, out_dir / "artifact_payload.pt")
        with open(out_dir / "submission_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"val_loss": vl, "val_bpb_sliding": vb_sl, "artifact_bytes": total_b}, f, indent=2)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
