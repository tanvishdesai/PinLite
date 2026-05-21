"""
Frozen source snapshot for GAQ research.

Source copied from:
  - PinPoint-main.py (AudioFeatureExtractor, get_sinusoidal_embeddings,
    GatedCrossAttentionBlock, PinpointTransformer)

Purpose:
  - Keep a stable local copy of the original full-model gated attention path
    used as the quantization target for the GAQ track.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_mfcc, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(128)
        self.gru = nn.GRU(input_size=128, hidden_size=embed_dim, batch_first=True)
        print("Initialized AudioFeatureExtractor with a CNN-GRU backbone. (Using LayerNorm instead of BatchNorm)")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.ln(x)
        output, _ = self.gru(x)
        return output


def get_sinusoidal_embeddings(n_position, d_hid):
    """Sinusoidal position encoding table."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_attn_output, cross_attn_map = self.audio_to_video_attn(
            query=audio_norm,
            key=video_norm,
            value=video_norm,
            key_padding_mask=video_mask,
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


class PinpointTransformer(nn.Module):
    """
    Original full-model forward logic snapshot.

    NOTE:
      This class expects a `config` object and a `VideoFeatureExtractor` class
      matching the original project. Kept here as source reference for GAQ.
    """

    def __init__(self, config, VideoFeatureExtractor):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractor(config.EMBED_DIM)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        self.gated_attention_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        num_offset_classes = 2 * config.MAX_OFFSET + 1
        self.offset_head = nn.Linear(config.EMBED_DIM, num_offset_classes)
        print("PinpointTransformer model initialized.")

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)

        if self.training and self.config.MODALITY_DROPOUT_PROB > 0:
            video_drop_mask = torch.ones(video_feat.size(0), 1, 1, device=video_feat.device)
            audio_drop_mask = torch.ones(audio_feat.size(0), 1, 1, device=audio_feat.device)
            for i in range(video.size(0)):
                if random.random() < self.config.MODALITY_DROPOUT_PROB:
                    if random.random() < 0.5:
                        video_drop_mask[i] = 0
                    else:
                        audio_drop_mask[i] = 0
            video_feat = video_feat * video_drop_mask
            audio_feat = audio_feat * audio_drop_mask

        video_feat = video_feat + self.video_pos_encoder[:, : video_feat.size(1), :]
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
