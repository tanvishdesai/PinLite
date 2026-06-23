"""
====================================================================================
GAQ Core — Quantization of Gated Cross-Attention (the technical nucleus of PIN-Lite v2)
====================================================================================
This module is the spine of the elevated project. It implements:

  1. Fake-quantization primitives (per-channel weights, per-tensor activations,
     min/max or percentile observers for outlier handling) with a straight-through
     estimator so the same code path supports BOTH PTQ and QAT.

  2. A *quantizable* explicit gated-cross-attention block. PyTorch's
     nn.MultiheadAttention cannot be quantized, so we decompose it into explicit
     Q/K/V/out-projection Linear ops. This is the prerequisite no prior multimodal
     forensic compression work has done for *gated* cross-modal attention.

  3. A hybrid-precision policy:
        - "fp32"  : nothing quantized (math-identical refactor; used for P1 parity)
        - "gaq"   : INT8 on Q/K/V/out-proj + gate-linear + FFN linears,
                    FP32 on softmax + LayerNorm + the sigmoid gate (the SAFE scheme)
        - "naive" : additionally quantizes softmax / LayerNorm / sigmoid outputs to
                    INT8 (the STRAWMAN that corrupts the attention map)

  4. Model surgery to swap the student's GatedCrossAttentionBlock for the
     quantizable version, calibration, and a QAT fine-tuning loop.

  5. Measurement instruments: EPS (Spearman + Top-k IoU of attention maps) and a
     *behavioral* faithfulness check (deletion / insertion on the attended video
     frames), plus accuracy/AUC/latency/size helpers.

Everything is written so it can run on CPU (for honest edge latency) or GPU.
====================================================================================
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)  # np.trapz removed in NumPy 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


# =================================================================================
# 0. REPRODUCIBILITY
# =================================================================================

def set_seed(seed: int = 11) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =================================================================================
# 1. FAKE-QUANTIZATION PRIMITIVES
# =================================================================================

class _FakeQuantSTE(torch.autograd.Function):
    """quantize -> dequantize with a straight-through gradient (identity)."""

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients to x only.
        return grad_output, None, None, None, None


def fake_quantize(x, scale, zero_point, qmin, qmax):
    return _FakeQuantSTE.apply(x, scale, zero_point, qmin, qmax)


def quantize_weight_per_channel(weight: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """
    Symmetric per-output-channel weight fake-quant. Weights are static so the
    scale is computed directly from the tensor (no observer needed).
    weight: [out_features, in_features]
    """
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax
    # per output channel (dim 0)
    w = weight
    max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    zero_point = torch.zeros_like(scale)
    return fake_quantize(w, scale, zero_point, qmin, qmax)


def quantize_weight_per_tensor(weight: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -qmax
    max_abs = weight.abs().max().clamp(min=1e-8)
    scale = max_abs / qmax
    return fake_quantize(weight, scale, torch.tensor(0.0, device=weight.device), qmin, qmax)


class ActFakeQuant(nn.Module):
    """
    Affine per-tensor activation fake-quant with a calibrated observer.

    During calibration (`self.calibrating == True`) it records running min/max
    (or a percentile estimate for outlier handling) and passes the float through.
    After `freeze()` it quantizes on every forward.
    """

    def __init__(self, num_bits: int = 8, observer: str = "minmax", percentile: float = 99.95):
        super().__init__()
        self.num_bits = num_bits
        self.observer = observer
        self.percentile = percentile
        self.calibrating = False
        self.enabled = False  # turned on after freeze()
        self.qmin = 0
        self.qmax = 2 ** num_bits - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))

    @torch.no_grad()
    def _observe(self, x: torch.Tensor) -> None:
        xf = x.detach().float().flatten()
        if self.observer == "percentile" and xf.numel() > 0:
            lo = torch.quantile(xf, 1.0 - self.percentile / 100.0)
            hi = torch.quantile(xf, self.percentile / 100.0)
        else:
            lo = xf.min()
            hi = xf.max()
        self.min_val = torch.minimum(self.min_val, lo)
        self.max_val = torch.maximum(self.max_val, hi)

    @torch.no_grad()
    def freeze(self) -> None:
        mn = float(self.min_val.item())
        mx = float(self.max_val.item())
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
            mn, mx = -1.0, 1.0
        # include zero in the range for affine quant
        mn = min(mn, 0.0)
        mx = max(mx, 0.0)
        scale = (mx - mn) / (self.qmax - self.qmin)
        scale = max(scale, 1e-8)
        zero_point = round(self.qmin - mn / scale)
        zero_point = float(min(self.qmax, max(self.qmin, zero_point)))
        self.scale = torch.tensor(scale)
        self.zero_point = torch.tensor(zero_point)
        self.calibrating = False
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._observe(x)
            return x
        if not self.enabled:
            return x
        scale = self.scale.to(x.device)
        zp = self.zero_point.to(x.device)
        return fake_quantize(x, scale, zp, self.qmin, self.qmax)


class QLinear(nn.Module):
    """
    A drop-in nn.Linear wrapper supporting INT8 fake-quant:
      - per-output-channel symmetric weight quant
      - per-tensor affine activation quant (calibrated)
    When `enabled` is False it is bit-exact to the wrapped nn.Linear (used for the
    FP32 parity baseline and for handing the inner nn.Linear to a real INT8 backend).
    """

    def __init__(self, linear: nn.Linear, num_bits: int = 8,
                 per_channel: bool = True, observer: str = "minmax"):
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.enabled = False
        self.act_fq = ActFakeQuant(num_bits=num_bits, observer=observer)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.linear(x)
        xq = self.act_fq(x)
        if self.per_channel:
            wq = quantize_weight_per_channel(self.linear.weight, self.num_bits)
        else:
            wq = quantize_weight_per_tensor(self.linear.weight, self.num_bits)
        return F.linear(xq, wq, self.linear.bias)


# =================================================================================
# 2. QUANTIZABLE EXPLICIT GATED CROSS-ATTENTION
# =================================================================================

def _qlinear_from_weight(weight: torch.Tensor, bias: Optional[torch.Tensor],
                         num_bits: int, per_channel: bool, observer: str) -> QLinear:
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=bias is not None)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    return QLinear(lin, num_bits=num_bits, per_channel=per_channel, observer=observer)


def _split_mha_into_qlinears(mha: nn.MultiheadAttention, num_bits: int,
                             per_channel: bool, observer: str
                             ) -> Tuple[QLinear, QLinear, QLinear, QLinear, int, int]:
    """Decompose nn.MultiheadAttention into explicit Q/K/V/out QLinear layers."""
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    w_q = W[:embed_dim, :]
    w_k = W[embed_dim:2 * embed_dim, :]
    w_v = W[2 * embed_dim:, :]
    b_q = b[:embed_dim] if b is not None else None
    b_k = b[embed_dim:2 * embed_dim] if b is not None else None
    b_v = b[2 * embed_dim:] if b is not None else None
    q = _qlinear_from_weight(w_q, b_q, num_bits, per_channel, observer)
    k = _qlinear_from_weight(w_k, b_k, num_bits, per_channel, observer)
    v = _qlinear_from_weight(w_v, b_v, num_bits, per_channel, observer)
    o = _qlinear_from_weight(mha.out_proj.weight, mha.out_proj.bias, num_bits, per_channel, observer)
    return q, k, v, o, embed_dim, num_heads


class QExplicitMHA(nn.Module):
    """Explicit multi-head attention (math-equivalent to nn.MultiheadAttention)."""

    def __init__(self, mha: nn.MultiheadAttention, num_bits: int,
                 per_channel: bool, observer: str):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.embed_dim, self.num_heads = \
            _split_mha_into_qlinears(mha, num_bits, per_channel, observer)
        self.head_dim = self.embed_dim // self.num_heads
        # optional fake-quant used ONLY by the naive strawman:
        #   - scores_fq quantizes the PRE-softmax attention scores, whose large
        #     dynamic range + outliers are exactly what makes naive INT8 softmax
        #     numerically fragile;
        #   - softmax_fq quantizes the softmax output.
        self.scores_fq = ActFakeQuant(num_bits=num_bits)
        self.softmax_fq = ActFakeQuant(num_bits=num_bits)
        self.quant_softmax = False

    def forward(self, query, key, value, key_padding_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.quant_softmax:                          # strawman: crush score range
            scores = self.scores_fq(scores)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)            # FP32 in the safe scheme
        if self.quant_softmax:                          # strawman also crushes probs
            attn = self.softmax_fq(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        mean_attn_map = attn.mean(dim=1)                # average over heads
        return out, mean_attn_map


class QGatedCrossAttentionBlock(nn.Module):
    """
    Quantizable, math-equivalent replacement for GatedCrossAttentionBlock.

    policy:
      "fp32"  -> bit-exact to the original block
      "gaq"   -> INT8 on all Linear ops; softmax/LN/sigmoid stay FP32
      "naive" -> additionally quantize softmax / LN / sigmoid outputs (strawman)
    """

    def __init__(self, original: nn.Module, num_bits: int = 8,
                 per_channel: bool = True, observer: str = "minmax"):
        super().__init__()
        self.num_bits = num_bits
        self.ln1 = copy.deepcopy(original.ln1)
        self.ln2 = copy.deepcopy(original.ln2)
        self.dropout = copy.deepcopy(original.dropout)

        self.cross_attn = QExplicitMHA(original.audio_to_video_attn, num_bits, per_channel, observer)
        self.self_attn = QExplicitMHA(original.self_attn, num_bits, per_channel, observer)

        # gate = Linear -> Sigmoid
        gate_lin = original.gate[0]
        self.gate_linear = _qlinear_from_weight(
            gate_lin.weight, gate_lin.bias, num_bits, per_channel, observer)

        # ffn = Linear -> GELU -> Dropout -> Linear
        ffn_lin1 = original.ffn[0]
        ffn_lin2 = original.ffn[3]
        self.ffn_linear1 = _qlinear_from_weight(
            ffn_lin1.weight, ffn_lin1.bias, num_bits, per_channel, observer)
        self.ffn_linear2 = _qlinear_from_weight(
            ffn_lin2.weight, ffn_lin2.bias, num_bits, per_channel, observer)
        self.ffn_act = copy.deepcopy(original.ffn[1])  # GELU
        self.ffn_drop = copy.deepcopy(original.ffn[2])

        # extra activation fake-quants used ONLY by the naive strawman
        self.ln1_fq = ActFakeQuant(num_bits=num_bits)
        self.ln2_fq = ActFakeQuant(num_bits=num_bits)
        self.gate_fq = ActFakeQuant(num_bits=num_bits)

        self.policy = "fp32"

    # -- policy plumbing -------------------------------------------------------
    def _all_qlinears(self) -> List[QLinear]:
        mods: List[QLinear] = []
        for m in self.modules():
            if isinstance(m, QLinear):
                mods.append(m)
        return mods

    def set_policy(self, policy: str) -> None:
        self.policy = policy
        quantize_linears = policy in ("gaq", "naive")
        for ql in self._all_qlinears():
            ql.enabled = quantize_linears
        quant_extra = policy == "naive"
        self.cross_attn.quant_softmax = quant_extra
        self.self_attn.quant_softmax = quant_extra
        for fq in (self.ln1_fq, self.ln2_fq, self.gate_fq):
            fq.enabled = quant_extra and fq.scale is not None

    def _maybe_q(self, fq: ActFakeQuant, x: torch.Tensor) -> torch.Tensor:
        if self.policy == "naive":
            return fq(x)
        return x

    def forward(self, audio_feat, video_feat, video_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_norm = self._maybe_q(self.ln1_fq, self.ln1(audio_feat))
        video_norm = self._maybe_q(self.ln1_fq, self.ln1(video_feat))

        cross_out, cross_map = self.cross_attn(
            audio_norm, video_norm, video_norm, key_padding_mask=video_mask)
        audio_feat = audio_feat + self.dropout(cross_out)

        gate = torch.sigmoid(self.gate_linear(audio_feat))
        gate = self._maybe_q(self.gate_fq, gate)
        gated = audio_feat * gate

        gated_norm = self._maybe_q(self.ln2_fq, self.ln2(gated))
        self_out, _ = self.self_attn(gated_norm, gated_norm, gated_norm)
        gated = gated + self.dropout(self_out)

        gated_norm2 = self._maybe_q(self.ln2_fq, self.ln2(gated))
        ffn = self.ffn_linear2(self.ffn_drop(self.ffn_act(self.ffn_linear1(gated_norm2))))
        out = gated + self.dropout(ffn)
        return out, cross_map


# =================================================================================
# 3. MODEL SURGERY
# =================================================================================

def build_quantizable_student(student: nn.Module, num_bits: int = 8,
                              per_channel: bool = True, observer: str = "minmax"
                              ) -> nn.Module:
    """
    Deep-copy the distilled student and replace each GatedCrossAttentionBlock with a
    QGatedCrossAttentionBlock. Returns an FP32 (policy='fp32') model that is
    math-equivalent to the original until a policy is set.
    """
    model = copy.deepcopy(student)
    new_layers = nn.ModuleList()
    for block in model.gated_attention_layers:
        new_layers.append(QGatedCrossAttentionBlock(
            block, num_bits=num_bits, per_channel=per_channel, observer=observer))
    model.gated_attention_layers = new_layers
    model.eval()
    return model


def set_model_policy(model: nn.Module, policy: str) -> None:
    for m in model.modules():
        if isinstance(m, QGatedCrossAttentionBlock):
            m.set_policy(policy)


def _all_act_fq(model: nn.Module) -> List[ActFakeQuant]:
    return [m for m in model.modules() if isinstance(m, ActFakeQuant)]


def start_calibration(model: nn.Module) -> None:
    for fq in _all_act_fq(model):
        fq.calibrating = True
        fq.enabled = False
        fq.min_val.fill_(float("inf"))
        fq.max_val.fill_(float("-inf"))


def finish_calibration(model: nn.Module) -> None:
    for fq in _all_act_fq(model):
        if fq.calibrating:
            fq.freeze()


@torch.no_grad()
def calibrate(model: nn.Module, dataloader, device: str, num_batches: int,
              policy: str = "gaq") -> None:
    """Run a few batches to populate activation observers, then freeze them."""
    model.eval().to(device)
    set_model_policy(model, policy)
    start_calibration(model)
    seen = 0
    for batch in dataloader:
        if batch is None:
            continue
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        model(video, audio)
        seen += 1
        if seen >= num_batches:
            break
    finish_calibration(model)
    # re-assert policy so the extra naive fake-quants pick up enabled flags
    set_model_policy(model, policy)
    print(f"  [calibrate] froze {len(_all_act_fq(model))} observers over {seen} batches "
          f"(policy={policy})")


# =================================================================================
# 4. QAT FINE-TUNING (recovers PTQ loss)
# =================================================================================

def qat_finetune(quant_model: nn.Module, teacher: nn.Module, dataloader, device: str,
                 epochs: int, lr: float, kd_alpha: float = 0.5, kd_beta: float = 5.0,
                 kd_temperature: float = 2.0, policy: str = "gaq",
                 max_batches: Optional[int] = None) -> nn.Module:
    """
    Quantization-aware training: the fake-quant nodes stay active (STE) while we
    fine-tune against the frozen FP32 teacher with the same KD objective used in
    distillation (hard BCE + soft KL + attention MSE).
    """
    set_model_policy(quant_model, policy)
    quant_model.train().to(device)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW([p for p in quant_model.parameters() if p.requires_grad], lr=lr)
    bce = nn.BCEWithLogitsLoss()
    T = kd_temperature

    for epoch in range(epochs):
        running = 0.0
        nb = 0
        for batch in dataloader:
            if batch is None:
                continue
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                t_logits, _, t_map = teacher(video, audio)
            s_logits, _, s_map = quant_model(video, audio)
            hard = bce(s_logits.squeeze(1), labels)
            soft = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1)) * (T * T)
            if s_map is not None and t_map is not None:
                m = min(s_map.shape[-1], t_map.shape[-1])
                a = min(s_map.shape[-2], t_map.shape[-2])
                attn = F.mse_loss(s_map[:, :a, :m], t_map[:, :a, :m])
            else:
                attn = torch.tensor(0.0, device=device)
            loss = kd_alpha * hard + (1 - kd_alpha) * soft + kd_beta * attn
            if not torch.isfinite(loss):
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(quant_model.parameters(), 1.0)
            optim.step()
            running += float(loss.item())
            nb += 1
            if max_batches is not None and nb >= max_batches:
                break
        print(f"  [QAT] epoch {epoch + 1}/{epochs}  avg_loss={running / max(nb, 1):.4f}")
    quant_model.eval()
    set_model_policy(quant_model, policy)
    return quant_model


# =================================================================================
# 5. REAL INT8 BACKEND PATH (for honest CPU latency + size)
# =================================================================================

def to_dynamic_int8(explicit_model: nn.Module) -> nn.Module:
    """
    Take a quantizable explicit student (fake-quant disabled) and apply PyTorch
    dynamic INT8 to its underlying nn.Linear and nn.GRU modules. Because attention
    is now explicit Linears, this reaches the attention projections that the
    original opaque nn.MultiheadAttention blocked — i.e. a *real* INT8 deployment of
    GAQ on a CPU backend (fbgemm/qnnpack). Softmax/LN/sigmoid stay FP32 naturally.
    """
    model = copy.deepcopy(explicit_model).cpu()
    set_model_policy(model, "fp32")  # disable fake-quant; use real backend instead
    qmodel = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    qmodel.eval()
    return qmodel


# =================================================================================
# 6. MEASUREMENT — accuracy / AUC
# =================================================================================

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader, device: str,
                      max_batches: Optional[int] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    model.eval().to(device)
    probs: List[float] = []
    labels: List[float] = []
    nb = 0
    for batch in dataloader:
        if batch is None:
            continue
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        logits, _, _ = model(video, audio)
        p = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        probs.extend(np.atleast_1d(p).tolist())
        labels.extend(batch["label"].numpy().tolist())
        nb += 1
        if max_batches is not None and nb >= max_batches:
            break
    if not labels:
        return {}
    preds = [1.0 if x > 0.5 else 0.0 for x in probs]
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0
    except Exception:
        out["auc"] = 0.0
    return out


# =================================================================================
# 7. MEASUREMENT — EPS (attention map fidelity)
# =================================================================================

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if spearmanr is None:
        # Pearson on ranks fallback
        ar = np.argsort(np.argsort(a)).astype(float)
        br = np.argsort(np.argsort(b)).astype(float)
        if ar.std() < 1e-9 or br.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(ar, br)[0, 1])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    c, _ = spearmanr(a, b)
    return 0.0 if np.isnan(c) else float(c)


def _topk_iou(a: np.ndarray, b: np.ndarray, percentile: float = 80.0) -> float:
    ta = np.percentile(a, percentile)
    tb = np.percentile(b, percentile)
    ma = a > ta
    mb = b > tb
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union > 0 else 0.0


@torch.no_grad()
def _get_attention_map(model: nn.Module, video, audio) -> Optional[np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    _, _, attn = model(video.to(dev), audio.to(dev))
    if attn is None:
        return None
    return attn.detach().cpu().float().numpy()


@torch.no_grad()
def compute_eps(teacher: nn.Module, student: nn.Module, dataloader, device: str,
                num_samples: int, w1: float = 0.5) -> Dict[str, float]:
    """EPS = w1 * Spearman(A_T, A_S) + w2 * TopK-IoU(A_T, A_S), averaged over samples."""
    teacher.eval()
    student.eval()
    spears: List[float] = []
    ious: List[float] = []
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        if batch is None:
            continue
        video, audio = batch["video"], batch["audio"]
        at = _get_attention_map(teacher, video, audio)
        as_ = _get_attention_map(student, video, audio)
        if at is None or as_ is None:
            continue
        bs = video.size(0)
        for i in range(bs):
            if count >= num_samples:
                break
            mt = at[i].flatten()
            ms = as_[i].flatten()
            n = min(len(mt), len(ms))
            mt, ms = mt[:n], ms[:n]
            spears.append(_spearman(mt, ms))
            ious.append(_topk_iou(mt, ms))
            count += 1
    if count == 0:
        return {}
    sp = float(np.mean(spears))
    iou = float(np.mean(ious))
    per_sample_eps = [w1 * s + (1 - w1) * j for s, j in zip(spears, ious)]
    return {
        "eps": w1 * sp + (1 - w1) * iou,
        "spearman": sp,
        "iou": iou,
        "eps_samples": count,
        # Per-sample arrays so callers can inspect the DISTRIBUTION (the mean can
        # wash out a tail divergence between naive and GAQ even when it is real).
        "per_sample_eps": per_sample_eps,
        "per_sample_spearman": spears,
        "per_sample_iou": ious,
        # Tail summary: the worst 10% of samples is where naive quant first shows.
        "eps_p10": float(np.percentile(per_sample_eps, 10)),
        "eps_p50": float(np.percentile(per_sample_eps, 50)),
        "spearman_p10": float(np.percentile(spears, 10)),
    }


# =================================================================================
# 8. MEASUREMENT — behavioral faithfulness (deletion / insertion)
# =================================================================================

@torch.no_grad()
def _video_frame_saliency(attn_map: torch.Tensor) -> torch.Tensor:
    """attn_map [B, Ta, Tv] -> per-video-frame saliency [B, Tv] (attention received)."""
    return attn_map.sum(dim=1)


@torch.no_grad()
def deletion_insertion_scores(model: nn.Module, dataloader, device: str,
                              num_samples: int, steps: int = 6
                              ) -> Dict[str, object]:
    """
    Behavioral faithfulness grounded in causality, not just map correlation.

    For each sample we rank video frames by attention received and (a) DELETE them
    most-salient-first, (b) INSERT them most-salient-first into a blanked clip,
    recording the model's fake-probability at each step. A faithful explanation
    yields a steep deletion curve (prob falls fast) and a steep insertion curve
    (prob rises fast). We return the mean AUC of each curve plus the per-sample
    deletion scores (used to measure teacher/student agreement).
    """
    model.eval().to(device)
    del_aucs: List[float] = []
    ins_aucs: List[float] = []
    per_sample_del: List[float] = []
    count = 0
    fractions = np.linspace(0.0, 1.0, steps + 1)
    for batch in dataloader:
        if count >= num_samples:
            break
        if batch is None:
            continue
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        logits, _, attn = model(video, audio)
        if attn is None:
            continue
        sal = _video_frame_saliency(attn)  # [B, Tv]
        B, Tv = sal.shape
        order = torch.argsort(sal, dim=1, descending=True)  # most salient first
        for i in range(B):
            if count >= num_samples:
                break
            v = video[i:i + 1]
            a = audio[i:i + 1]
            ranked = order[i]
            del_curve = []
            ins_curve = []
            for f in fractions:
                k = int(round(f * Tv))
                # deletion: zero the top-k salient frames
                vd = v.clone()
                if k > 0:
                    vd[:, ranked[:k], :, :, :] = 0.0
                pd = torch.sigmoid(model(vd, a)[0].float()).item()
                del_curve.append(pd)
                # insertion: start from blank, add back top-k salient frames
                vi = torch.zeros_like(v)
                if k > 0:
                    vi[:, ranked[:k], :, :, :] = v[:, ranked[:k], :, :, :]
                pi = torch.sigmoid(model(vi, a)[0].float()).item()
                ins_curve.append(pi)
            del_auc = float(_trapz(del_curve, fractions))
            ins_auc = float(_trapz(ins_curve, fractions))
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)
            per_sample_del.append(del_auc)
            count += 1
    if count == 0:
        return {}
    return {
        "deletion_auc": float(np.mean(del_aucs)),
        "insertion_auc": float(np.mean(ins_aucs)),
        "faith_samples": count,
        "per_sample_deletion": per_sample_del,
    }


def faithfulness_agreement(teacher_del: List[float], student_del: List[float]) -> float:
    """Spearman agreement of per-sample deletion scores between teacher and student."""
    n = min(len(teacher_del), len(student_del))
    if n < 3:
        return 0.0
    return _spearman(np.array(teacher_del[:n]), np.array(student_del[:n]))


# =================================================================================
# 9. MEASUREMENT — latency & size
# =================================================================================

@torch.no_grad()
def measure_cpu_latency(model: nn.Module, sample_video: torch.Tensor,
                        sample_audio: torch.Tensor, repeats: int, warmup: int) -> float:
    """Mean single-sample CPU latency in ms."""
    model.eval().cpu()
    v = sample_video[:1].cpu()
    a = sample_audio[:1].cpu()
    for _ in range(warmup):
        model(v, a)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(v, a)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_int8_size_mb(model: nn.Module) -> float:
    """
    Theoretical on-disk size if every QLinear (when enabled) and nn.GRU is stored as
    INT8 weights and everything else as FP32. This reflects what the real-backend
    export achieves, while the fake-quant model on disk would still look like FP32.
    """
    int8_params = 0
    fp32_params = 0
    quantized_weight_ids = set()
    for m in model.modules():
        if isinstance(m, QLinear) and m.enabled:
            quantized_weight_ids.add(id(m.linear.weight))
        if isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                quantized_weight_ids.add(id(p))
    for p in model.parameters():
        if id(p) in quantized_weight_ids:
            int8_params += p.numel()
        else:
            fp32_params += p.numel()
    bytes_total = int8_params * 1 + fp32_params * 4
    return bytes_total / (1024 * 1024)


def actual_state_dict_size_mb(model: nn.Module, tmp_path: str = "_tmp_size.pth") -> float:
    torch.save(model.state_dict(), tmp_path)
    size = os.path.getsize(tmp_path) / (1024 * 1024)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return size


# =================================================================================
# 10. MODEL LOADERS (teacher + student) via gaq_config
# =================================================================================

def load_teacher(device: str = "cpu"):
    import gaq_config as C
    PinPoint = C.load_pinpoint()
    cfg = PinPoint.Config()
    C.apply_data_paths(cfg)
    model = PinPoint.PinpointTransformer(cfg)
    state = torch.load(C.TEACHER_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def load_student(device: str = "cpu"):
    import gaq_config as C
    Distill = C.load_distill()
    cfg = Distill.ConfigLite()
    C.apply_data_paths(cfg)
    model = Distill.PinpointTransformerLite(cfg)
    state = torch.load(C.STUDENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, cfg


def make_loaders(cfg, splits=("test",), batch_size: int = 8, shuffle: bool = False,
                 num_workers: int = 2):
    import gaq_config as C
    PinPoint = C.load_pinpoint()
    from torch.utils.data import DataLoader
    loaders = {}
    for split in splits:
        ds = PinPoint.LAVDFDataset(cfg, split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=PinPoint.collate_fn, num_workers=num_workers)
    return loaders


def get_one_batch(dataloader):
    for batch in dataloader:
        if batch is not None:
            return batch
    raise RuntimeError("No valid batch found in dataloader.")


# =================================================================================
# 11. RESULTS LOGGING (shared schema for the Table 1 builder)
# =================================================================================

RESULT_FIELDS = [
    "model", "acc", "f1", "auc", "eps", "spearman", "iou",
    "deletion_auc", "insertion_auc", "faith_agreement",
    "latency_cpu_ms", "size_mb", "params_m", "speedup_vs_teacher", "source_note",
]


def append_result(csv_path: str, row: Dict[str, object]) -> None:
    import csv
    exists = os.path.exists(csv_path)
    full = {k: row.get(k, "") for k in RESULT_FIELDS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(full)
    print(f"  [log] appended '{row.get('model')}' -> {csv_path}")
