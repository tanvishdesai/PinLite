"""
Frozen source snapshot for GAQ research.

Source copied from:
  - Distill-student.py (ConfigLite, VideoFeatureExtractorLite, PinpointTransformerLite)

Purpose:
  - Keep a stable local copy of the PinLite student architecture that consumes
    the original GatedCrossAttentionBlock.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from snapshot_pinpoint_gated_attention import (
    AudioFeatureExtractor,
    GatedCrossAttentionBlock,
    get_sinusoidal_embeddings,
)


class ConfigLite:
    """Snapshot of student config used in PinLite."""

    # Student architecture
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.15

    # Student training
    EPOCHS = 20
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8

    # Distillation params
    KD_ALPHA = 0.5
    KD_BETA = 5.0
    KD_TEMPERATURE = 2.0

    # Core model constants needed for forward path
    NUM_MFCC = 13
    NUM_FRAMES = 30
    VIDEO_SIZE = (128, 128)
    MAX_OFFSET = 5


class VideoFeatureExtractorLite(nn.Module):
    """A lightweight video feature extractor using MobileNetV3-Small."""

    def __init__(self, embed_dim):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        mobilenet_out_features = 576
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(mobilenet_out_features, embed_dim)
        for param in self.feature_extractor[:3].parameters():
            param.requires_grad = False
        print("Initialized VideoFeatureExtractorLite with a pretrained MobileNetV3-Small backbone.")

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x)
        pooled_features = self.pool(features).view(b * t, -1)
        projected_features = self.projection(pooled_features)
        output = projected_features.view(b, t, -1)
        return output


class PinpointTransformerLite(nn.Module):
    """Snapshot of PinLite student model with original gated attention block."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractorLite(config.EMBED_DIM)
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
        print("PinpointTransformerLite (Student Model) initialized.")

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)
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
