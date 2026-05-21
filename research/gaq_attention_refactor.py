"""
Quantized Gated-Attention research utility:
- explicit Q/K/V attention math using weights from nn.MultiheadAttention
- parity harness against original GatedCrossAttentionBlock from PinPoint-main.py
"""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_pinpoint_main_module() -> object:
    root = Path(__file__).resolve().parent.parent
    file_path = root / "PinPoint-main.py"
    if not file_path.exists():
        raise FileNotFoundError(f"PinPoint-main.py not found at {file_path}")

    spec = importlib.util.spec_from_file_location("pinpoint_main_local", str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for PinPoint-main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_qkv_weights(mha: nn.MultiheadAttention) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mha.in_proj_weight is None:
        raise ValueError("Expected in_proj_weight for same-dim QKV projection.")

    embed_dim = mha.embed_dim
    w_q = mha.in_proj_weight[:embed_dim, :]
    w_k = mha.in_proj_weight[embed_dim : 2 * embed_dim, :]
    w_v = mha.in_proj_weight[2 * embed_dim :, :]

    b_q = b_k = b_v = None
    if mha.in_proj_bias is not None:
        b_q = mha.in_proj_bias[:embed_dim]
        b_k = mha.in_proj_bias[embed_dim : 2 * embed_dim]
        b_v = mha.in_proj_bias[2 * embed_dim :]

    return w_q, w_k, w_v, b_q, b_k, b_v


def explicit_multihead_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mha: nn.MultiheadAttention,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Explicit scaled dot-product attention using MHA weights.
    Returns:
      - attn_output: [B, Tq, E]
      - mean_attn_map: [B, Tq, Tk]
    """
    if not mha.batch_first:
        raise ValueError("This utility expects batch_first=True.")

    bsz, tgt_len, embed_dim = query.shape
    src_len = key.shape[1]
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError("embed_dim must be divisible by num_heads.")

    w_q, w_k, w_v, b_q, b_k, b_v = _split_qkv_weights(mha)

    q = F.linear(query, w_q, b_q)
    k = F.linear(key, w_k, b_k)
    v = F.linear(value, w_v, b_v)

    q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, src_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if key_padding_mask is not None:
        # expected shape [B, Tk] with True for masked positions
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    if mha.dropout > 0.0 and mha.training:
        attn_weights = F.dropout(attn_weights, p=mha.dropout, training=True)

    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
    attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)

    mean_attn_map = attn_weights.mean(dim=1)
    return attn_output, mean_attn_map


class ExplicitGatedCrossAttentionBlock(nn.Module):
    """
    Math-equivalent refactor of GatedCrossAttentionBlock:
    keeps all original parameters but replaces MHA forward calls
    with explicit Q/K/V + softmax + weighted sum math.
    """

    def __init__(self, original_block: nn.Module):
        super().__init__()
        self.ln1 = copy.deepcopy(original_block.ln1)
        self.ln2 = copy.deepcopy(original_block.ln2)
        self.audio_to_video_attn = copy.deepcopy(original_block.audio_to_video_attn)
        self.gate = copy.deepcopy(original_block.gate)
        self.self_attn = copy.deepcopy(original_block.self_attn)
        self.ffn = copy.deepcopy(original_block.ffn)
        self.dropout = copy.deepcopy(original_block.dropout)

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)

        cross_attn_output, cross_attn_map = explicit_multihead_attention(
            query=audio_norm,
            key=video_norm,
            value=video_norm,
            mha=self.audio_to_video_attn,
            key_padding_mask=video_mask,
        )
        audio_feat = audio_feat + self.dropout(cross_attn_output)

        gated_audio_feat = audio_feat * self.gate(audio_feat)
        gated_audio_norm = self.ln2(gated_audio_feat)

        self_attn_output, _ = explicit_multihead_attention(
            query=gated_audio_norm,
            key=gated_audio_norm,
            value=gated_audio_norm,
            mha=self.self_attn,
            key_padding_mask=None,
        )
        gated_audio_feat = gated_audio_feat + self.dropout(self_attn_output)

        gated_audio_norm2 = self.ln2(gated_audio_feat)
        ffn_output = self.ffn(gated_audio_norm2)
        final_output = gated_audio_feat + self.dropout(ffn_output)
        return final_output, cross_attn_map


def run_block_parity(
    seed: int = 11,
    trials: int = 8,
    embed_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.0,
    atol: float = 2e-5,
) -> Dict[str, float]:
    """
    Compares original vs explicit block outputs on random masked inputs.
    Returns dict with mean/max absolute errors and pass flag.
    """
    torch.manual_seed(seed)

    module = _load_pinpoint_main_module()
    if not hasattr(module, "GatedCrossAttentionBlock"):
        raise AttributeError("GatedCrossAttentionBlock not found in PinPoint-main.py")

    original = module.GatedCrossAttentionBlock(embed_dim, num_heads, dropout)
    explicit = ExplicitGatedCrossAttentionBlock(original)
    original.eval()
    explicit.eval()

    out_diffs = []
    attn_diffs = []

    with torch.no_grad():
        for idx in range(trials):
            bsz = 2 + (idx % 2)
            audio_len = 60 + idx * 3
            video_len = 20 + idx

            audio_feat = torch.randn(bsz, audio_len, embed_dim)
            video_feat = torch.randn(bsz, video_len, embed_dim)

            mask = torch.zeros(bsz, video_len, dtype=torch.bool)
            if video_len > 4:
                mask[0, -2:] = True

            out_ref, attn_ref = original(audio_feat, video_feat, mask)
            out_new, attn_new = explicit(audio_feat, video_feat, mask)

            out_diff = torch.abs(out_ref - out_new)
            attn_diff = torch.abs(attn_ref - attn_new)
            out_diffs.append(out_diff)
            attn_diffs.append(attn_diff)

    all_out = torch.cat([t.flatten() for t in out_diffs], dim=0)
    all_attn = torch.cat([t.flatten() for t in attn_diffs], dim=0)

    mean_abs_output = float(all_out.mean().item())
    max_abs_output = float(all_out.max().item())
    mean_abs_attn = float(all_attn.mean().item())
    max_abs_attn = float(all_attn.max().item())

    parity_pass = (max_abs_output <= atol) and (max_abs_attn <= atol)
    return {
        "parity_pass": bool(parity_pass),
        "mean_abs_output": mean_abs_output,
        "max_abs_output": max_abs_output,
        "mean_abs_attn": mean_abs_attn,
        "max_abs_attn": max_abs_attn,
    }

