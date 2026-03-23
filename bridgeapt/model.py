import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GVP(nn.Module):
    """Geometric Vector Perceptron 模块"""
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.vec_proj = nn.Linear(21, 32)
        self.scalar_proj = nn.Sequential(
            nn.Linear(node_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, coords, scalar_features):
        vec_features = self.vec_proj(coords)
        scalar_features = self.scalar_proj(scalar_features)
        return vec_features, scalar_features


class BridgeAPT(nn.Module):
    """GVP + Transformer 核酸序列生成模型"""
    def __init__(self, vocab_size=5, d_model=512, nhead=4, num_layers=6):
        super().__init__()
        self.gvp = GVP(node_dim=6, edge_dim=32)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.projection = nn.Linear(544, d_model)

    def forward(self, coords, scalar_features, src_mask=None):
        vec_feats, scalar_feats = self.gvp(coords, scalar_features)
        combined = torch.cat([vec_feats, scalar_feats], dim=-1)
        projected = self.projection(combined)
        output = self.transformer_encoder(projected.transpose(0, 1), src_mask)
        logits = self.decoder(output.transpose(0, 1))
        return logits
