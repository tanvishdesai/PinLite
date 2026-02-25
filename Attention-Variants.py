"""
====================================================================================
PIN-Lite: Attention Mechanism Variants (Phase B — Attention Compression)
====================================================================================
Defines three alternative attention blocks that are DROP-IN REPLACEMENTS for 
the original GatedCrossAttentionBlock used in PinPoint/PinPointLite:

1. LinearCrossAttentionBlock  — O(n) linear attention via ELU+1 kernel
2. MQACrossAttentionBlock     — Multi-Query Attention (shared K/V projections)
3. LowRankCrossAttentionBlock — Low-rank Q/K/V factorization via bottleneck

Each block also has a corresponding PinpointTransformer variant and Config.

Usage:
    This file is imported by Distill-Attention-Variants.py for training,
    and by Comprehensive-Testing-V3.py for evaluation.
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# =================================================================================
# IMPORTS FROM BASE
# =================================================================================
try:
    from PinPoint import (
        Config as TeacherConfig,
        AudioFeatureExtractor,
        get_sinusoidal_embeddings,
    )
    print("✅ [Attention-Variants] Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found.")
    sys.exit(1)

try:
    from Distill import VideoFeatureExtractorLite
    print("✅ [Attention-Variants] Loaded Distill_PinPoint module")
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Distill_PinPoint",
            os.path.join(os.path.dirname(__file__), "Distill-student.py"))
        DP = importlib.util.module_from_spec(spec)
        sys.modules["Distill_PinPoint"] = DP
        spec.loader.exec_module(DP)
        VideoFeatureExtractorLite = DP.VideoFeatureExtractorLite
        print("✅ [Attention-Variants] Loaded Distill-student via file path")
    except Exception as e:
        print(f"FATAL: Could not load Distill-student: {e}")
        sys.exit(1)


# =================================================================================
# VARIANT 1: LINEAR ATTENTION (O(n) complexity)
# =================================================================================
# Replaces softmax(Q·K^T)·V with φ(Q)·(φ(K)^T·V), where φ = ELU + 1.
# This avoids the N×N attention matrix, reducing complexity from O(n²) to O(n·d).
# Reference: Katharopoulos et al., "Transformers are RNNs" (2020)
# =================================================================================

class LinearAttention(nn.Module):
    """
    Linear attention using ELU+1 feature map.
    Complexity: O(n·d) instead of O(n²·d).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _elu_feature_map(self, x):
        """
        ELU + 1 feature map: ensures positive values for kernel trick.
        Added eps to prevent exact 0s which cause division-by-zero NaNs.
        """
        return F.elu(x) + 1.0 + 1e-5
    
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        # Project and roughly clamp to prevent extreme outliers before ELU
        Q = self.q_proj(query).clamp(-10.0, 10.0).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).clamp(-10.0, 10.0).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).clamp(-10.0, 10.0).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply numerical-safe feature map
        Q = self._elu_feature_map(Q)
        K = self._elu_feature_map(K)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (batch, k_len) — True means masked
            mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, k_len, 1)
            K = K.masked_fill(mask, 1e-5) # Fill with eps instead of exactly 0
            V = V.masked_fill(mask, 0.0)
        
        # Cast to float32 to prevent FP16 overflow during large einsum summations
        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()
        
        # Linear attention: φ(Q) · (φ(K)^T · V) 
        # Instead of softmax(Q · K^T) · V
        KV = torch.einsum('bhnd,bhne->bhde', K_f, V_f)  # (batch, heads, head_dim, head_dim)
        
        # Calculate normalizer with strict clamping to avoid div-by-zero
        Z_denom = torch.einsum('bhnd,bhd->bhn', Q_f, K_f.sum(dim=2))
        Z = 1.0 / (Z_denom.clamp_min(1e-5))
        
        attn_output = torch.einsum('bhnd,bhde,bhn->bhne', Q_f, KV, Z)
        
        # Safety clamp before casting back down to float16 to prevent overflow
        attn_output = attn_output.clamp(-65000.0, 65000.0).type_as(query)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Generate a pseudo attention map for compatibility (approximate)
        # We create a simplified map for EPS/distillation compatibility
        with torch.no_grad():
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', Q_f, K_f)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-5)
            attn_map = attn_weights.mean(dim=1).type_as(query)  # Average over heads
        
        return attn_output, attn_map


class LinearCrossAttentionBlock(nn.Module):
    """
    Drop-in replacement for GatedCrossAttentionBlock using Linear Attention.
    Same interface: forward(audio_feat, video_feat, video_mask) -> (output, attn_map)
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = LinearAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_attn_output, cross_attn_map = self.audio_to_video_attn(
            query=audio_norm, key=video_norm, value=video_norm, key_padding_mask=video_mask
        )
        audio_feat = audio_feat + self.dropout(cross_attn_output)
        gated_audio_feat = audio_feat * self.gate(audio_feat)
        gated_audio_norm = self.ln2(gated_audio_feat)
        self_attn_output, _ = self.self_attn(gated_audio_norm, gated_audio_norm, gated_audio_norm)
        gated_audio_feat = gated_audio_feat + self.dropout(self_attn_output)
        gated_audio_norm2 = self.ln2(gated_audio_feat)
        ffn_output = self.ffn(gated_audio_norm2)
        final_output = gated_audio_feat + self.dropout(ffn_output)
        return final_output, cross_attn_map


# =================================================================================
# VARIANT 2: MULTI-QUERY ATTENTION (MQA)
# =================================================================================
# Shares K and V projections across all heads (only Q is per-head).
# Reduces parameter count for K/V from num_heads * head_dim * embed_dim to just
# head_dim * embed_dim — ~40% parameter reduction in attention layers.
# Reference: Shazeer, "Fast Transformer Decoding" (2019)
# =================================================================================

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: separate Q per head, shared K/V across all heads.
    Reduces KV parameter count by num_heads×.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        # Q: separate projection per head (standard)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # K, V: shared across all heads (MQA key insight)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)  # Single head dim
        self.v_proj = nn.Linear(embed_dim, self.head_dim)  # Single head dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        # Q: per-head
        Q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: shared — expand to all heads
        K = self.k_proj(key).unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, k_len, head_dim)
        V = self.v_proj(value).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Standard scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, q_len, k_len)
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, H, q_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Return mean attention map across heads for compatibility
        attn_map = attn_weights.mean(dim=1)  # (B, q_len, k_len)
        
        return attn_output, attn_map


class MQACrossAttentionBlock(nn.Module):
    """
    Drop-in replacement for GatedCrossAttentionBlock using Multi-Query Attention.
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = MultiQueryAttention(embed_dim, num_heads, dropout)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = MultiQueryAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_attn_output, cross_attn_map = self.audio_to_video_attn(
            query=audio_norm, key=video_norm, value=video_norm, key_padding_mask=video_mask
        )
        audio_feat = audio_feat + self.dropout(cross_attn_output)
        gated_audio_feat = audio_feat * self.gate(audio_feat)
        gated_audio_norm = self.ln2(gated_audio_feat)
        self_attn_output, _ = self.self_attn(gated_audio_norm, gated_audio_norm, gated_audio_norm)
        gated_audio_feat = gated_audio_feat + self.dropout(self_attn_output)
        gated_audio_norm2 = self.ln2(gated_audio_feat)
        ffn_output = self.ffn(gated_audio_norm2)
        final_output = gated_audio_feat + self.dropout(ffn_output)
        return final_output, cross_attn_map


# =================================================================================
# VARIANT 3: LOW-RANK ATTENTION
# =================================================================================
# Decomposes Q/K/V projection matrices via a bottleneck layer:
#   W_q = W1_q · W2_q where W1 ∈ R^{d×r} and W2 ∈ R^{r×d} with r << d
# This reduces parameters from d² to 2·d·r per projection.
# Inspired by LoRA (Hu et al., 2021) and SVD-based compression.
# =================================================================================

class LowRankAttention(nn.Module):
    """
    Low-Rank factorized attention.
    Q/K/V projections go through a bottleneck of rank r = embed_dim // rank_factor.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, rank_factor=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.rank = embed_dim // rank_factor  # Bottleneck dimension
        
        # Low-rank Q: d → r → d
        self.q_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, embed_dim, bias=True)
        
        # Low-rank K: d → r → d
        self.k_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.k_up = nn.Linear(self.rank, embed_dim, bias=True)
        
        # Low-rank V: d → r → d
        self.v_down = nn.Linear(embed_dim, self.rank, bias=False)
        self.v_up = nn.Linear(self.rank, embed_dim, bias=True)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        # Low-rank projections
        Q = self.q_up(self.q_down(query))
        K = self.k_up(self.k_down(key))
        V = self.v_up(self.v_down(value))
        
        # Reshape for multi-head
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        attn_map = attn_weights.mean(dim=1)
        return attn_output, attn_map


class LowRankCrossAttentionBlock(nn.Module):
    """
    Drop-in replacement for GatedCrossAttentionBlock using Low-Rank Attention.
    """
    def __init__(self, embed_dim, num_heads, dropout, rank_factor=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = LowRankAttention(embed_dim, num_heads, dropout, rank_factor)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = LowRankAttention(embed_dim, num_heads, dropout, rank_factor)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_attn_output, cross_attn_map = self.audio_to_video_attn(
            query=audio_norm, key=video_norm, value=video_norm, key_padding_mask=video_mask
        )
        audio_feat = audio_feat + self.dropout(cross_attn_output)
        gated_audio_feat = audio_feat * self.gate(audio_feat)
        gated_audio_norm = self.ln2(gated_audio_feat)
        self_attn_output, _ = self.self_attn(gated_audio_norm, gated_audio_norm, gated_audio_norm)
        gated_audio_feat = gated_audio_feat + self.dropout(self_attn_output)
        gated_audio_norm2 = self.ln2(gated_audio_feat)
        ffn_output = self.ffn(gated_audio_norm2)
        final_output = gated_audio_feat + self.dropout(ffn_output)
        return final_output, cross_attn_map


# =================================================================================
# MODEL VARIANTS — Full student models using each attention type
# =================================================================================

class ConfigLinear(TeacherConfig):
    """Config for Linear Attention student."""
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.15
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    KD_ALPHA = 0.5
    KD_BETA = 5.0
    KD_TEMPERATURE = 2.0
    ATTENTION_TYPE = "linear"

class ConfigMQA(TeacherConfig):
    """Config for Multi-Query Attention student."""
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.15
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    KD_ALPHA = 0.5
    KD_BETA = 5.0
    KD_TEMPERATURE = 2.0
    ATTENTION_TYPE = "mqa"

class ConfigLowRank(TeacherConfig):
    """Config for Low-Rank Attention student."""
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.15
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    KD_ALPHA = 0.5
    KD_BETA = 5.0
    KD_TEMPERATURE = 2.0
    ATTENTION_TYPE = "lowrank"
    RANK_FACTOR = 4  # Bottleneck = embed_dim // rank_factor = 32


def get_attention_block(attention_type, embed_dim, num_heads, dropout, rank_factor=4):
    """Factory function to create the right attention block."""
    if attention_type == "linear":
        return LinearCrossAttentionBlock(embed_dim, num_heads, dropout)
    elif attention_type == "mqa":
        return MQACrossAttentionBlock(embed_dim, num_heads, dropout)
    elif attention_type == "lowrank":
        return LowRankCrossAttentionBlock(embed_dim, num_heads, dropout, rank_factor)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


class PinpointTransformerVariant(nn.Module):
    """
    Student model with configurable attention mechanism.
    Same architecture as PinpointTransformerLite but with swappable attention blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractorLite(config.EMBED_DIM)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        
        rank_factor = getattr(config, 'RANK_FACTOR', 4)
        self.gated_attention_layers = nn.ModuleList([
            get_attention_block(config.ATTENTION_TYPE, config.EMBED_DIM, config.NUM_HEADS, 
                              config.DROPOUT, rank_factor)
            for _ in range(config.NUM_LAYERS)
        ])
        
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        num_offset_classes = 2 * config.MAX_OFFSET + 1
        self.offset_head = nn.Linear(config.EMBED_DIM, num_offset_classes)
        print(f"PinpointTransformerVariant ({config.ATTENTION_TYPE}) initialized.")
    
    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)
        video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
        audio_len = audio_feat.size(1)
        audio_pos_encoding = get_sinusoidal_embeddings(audio_len, self.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos_encoding
        
        last_attention_map = None
        for layer in self.gated_attention_layers:
            audio_feat, attention_map = layer(audio_feat, video_feat, video_mask)
            last_attention_map = attention_map
        
        pooled_output = audio_feat.mean(dim=1)
        classification_logits = self.classification_head(pooled_output)
        offset_logits = self.offset_head(pooled_output)
        return classification_logits, offset_logits, last_attention_map


# =================================================================================
# PARAMETER COUNT COMPARISON
# =================================================================================

def compare_parameter_counts():
    """Print parameter counts for each attention variant vs the original."""
    from PinPoint import GatedCrossAttentionBlock
    
    embed_dim = 128
    num_heads = 4
    dropout = 0.15
    
    blocks = {
        "Original (GatedCrossAttention)": GatedCrossAttentionBlock(embed_dim, num_heads, dropout),
        "Linear Attention": LinearCrossAttentionBlock(embed_dim, num_heads, dropout),
        "Multi-Query Attention": MQACrossAttentionBlock(embed_dim, num_heads, dropout),
        "Low-Rank Attention (r=d/4)": LowRankCrossAttentionBlock(embed_dim, num_heads, dropout, rank_factor=4),
    }
    
    print("\n" + "="*60)
    print("ATTENTION BLOCK PARAMETER COMPARISON")
    print("="*60)
    print(f"{'Variant':<35} {'Params':>10} {'vs Original':>12}")
    print("-"*60)
    
    original_params = sum(p.numel() for p in blocks["Original (GatedCrossAttention)"].parameters())
    
    for name, block in blocks.items():
        params = sum(p.numel() for p in block.parameters())
        ratio = params / original_params
        print(f"{name:<35} {params:>10,} {ratio:>11.1%}")
    
    print()


if __name__ == "__main__":
    compare_parameter_counts()
    
    # Quick test: verify all variants produce correct output shapes
    print("\n--- Smoke Test: Forward Pass ---")
    batch_size = 2
    seq_len_audio = 100
    seq_len_video = 15
    embed_dim = 128
    num_heads = 4
    
    audio = torch.randn(batch_size, seq_len_audio, embed_dim)
    video = torch.randn(batch_size, seq_len_video, embed_dim)
    
    for name, BlockClass in [
        ("Linear", LinearCrossAttentionBlock),
        ("MQA", MQACrossAttentionBlock),
        ("LowRank", LowRankCrossAttentionBlock),
    ]:
        block = BlockClass(embed_dim, num_heads, 0.1)
        output, attn_map = block(audio, video)
        print(f"  {name}: output={output.shape}, attn_map={attn_map.shape} ✅")
    
    print("\nAll attention variants verified.")
