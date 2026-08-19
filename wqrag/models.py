"""
Stage II — Reconstruction-based detectors (Section 2.2.2; Fig. 2 Stage II).

PatchTST Autoencoder
    patch 16 steps (4 h) x 6 patches, d_model = 128, 8 heads, dropout 0.2,
    encoder 3 TransformerEncoder layers -> 64-d bottleneck ->
    decoder 2 TransformerEncoder layers -> Linear -> R^{96x5}

LSTM Autoencoder
    encoder 2-layer LSTM (128 hidden) -> repeated context ->
    decoder 2-layer LSTM (128 hidden) -> Linear -> R^{96x5}, dropout 0.2

Both expose `reconstruction_error(x)` returning the per-window MSE e(x).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, C) -> (B, N, d)
        b, t, c = x.shape
        n = t // self.patch_len
        x = x[:, : n * self.patch_len, :].reshape(b, n, self.patch_len * c)
        return self.proj(x)


class PatchTSTAutoencoder(nn.Module):
    """Patch-based Transformer autoencoder for multivariate windows."""

    def __init__(self, n_features: int, window_size: int = C.WINDOW_SIZE,
                 patch_len: int = C.PATCHTST["patch_len"], d_model: int = C.PATCHTST["d_model"],
                 n_heads: int = C.PATCHTST["n_heads"],
                 n_encoder_layers: int = C.PATCHTST["n_encoder_layers"],
                 n_decoder_layers: int = C.PATCHTST["n_decoder_layers"],
                 bottleneck_dim: int = C.PATCHTST["bottleneck_dim"],
                 dropout: float = C.PATCHTST["dropout"]):
        super().__init__()
        assert window_size % patch_len == 0, "window must be a multiple of patch length"
        self.n_features, self.window_size, self.patch_len = n_features, window_size, patch_len
        self.n_patches = window_size // patch_len

        self.embed = PatchEmbedding(n_features, patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        def _layer():
            return nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                              dim_feedforward=4 * d_model, dropout=dropout,
                                              batch_first=True, activation="gelu")

        self.encoder = nn.TransformerEncoder(_layer(), num_layers=n_encoder_layers)
        self.bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, d_model)
        self.decoder = nn.TransformerEncoder(_layer(), num_layers=n_decoder_layers)
        self.head = nn.Linear(d_model, patch_len * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.encoder(self.embed(x) + self.pos)
        h = self.expand(self.bottleneck(h))
        h = self.decoder(h)
        out = self.head(h).reshape(b, self.n_patches * self.patch_len, self.n_features)
        return out[:, : self.window_size, :]

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))


class LSTMAutoencoder(nn.Module):
    """Recurrent baseline (Malhotra et al., 2016)."""

    def __init__(self, n_features: int, hidden_dim: int = C.LSTM_AE["hidden_dim"],
                 n_layers: int = C.LSTM_AE["n_layers"], dropout: float = C.LSTM_AE["dropout"]):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        ctx = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)   # fixed-length code
        dec, _ = self.decoder(ctx)
        return self.head(dec)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))


MODEL_REGISTRY = {"PatchTST": PatchTSTAutoencoder, "LSTM-AE": LSTMAutoencoder}


def build_model(name: str, n_features: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_features=n_features)
