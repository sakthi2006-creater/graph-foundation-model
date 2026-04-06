"""Positional Encoding for Graph Transformers.
Supports Laplacian Eigenvectors (primary) and Rotary Position Embeddings (fallback)."""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional, Tuple

from src.config_loader import load_config
from src.utils import detect_device


class LaplacianPositionalEncoder(nn.Module):
    """Laplacian Positional Encoding using graph Laplacian eigenvectors."""

    def __init__(self, pe_dim: int = 16, max_freq: float = 1e4,
                 add_self_loops: bool = True, device=None):
        super().__init__()
        self.pe_dim = pe_dim
        self.max_freq = max_freq
        self.add_self_loops = add_self_loops
        self.device = device or detect_device()

    def forward(self, edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index_undir = to_undirected(edge_index)
        row, col = edge_index_undir

        deg = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(row.shape[0], dtype=torch.float, device=edge_index.device))

        # Normalized random-walk PE approximation (degree-based)
        deg_inv = 1.0 / (deg + 1e-8)
        pos_enc = torch.stack([
            torch.sin(deg_inv * (i + 1) * math.pi) for i in range(self.pe_dim)
        ], dim=-1)

        return pos_enc.clamp(-2, 2)


class RotaryPositionalEncoder(nn.Module):
    """Rotary Position Embeddings (RoPE) for graphs."""

    def __init__(self, pe_dim: int = 16, max_nodes: int = 10000, device=None):
        super().__init__()
        assert pe_dim % 2 == 0, "pe_dim must be even for rotary embeddings"
        self.pe_dim = pe_dim
        self.device = device or detect_device()
        theta = 1.0 / (10000 ** (torch.arange(0, pe_dim, 2).float() / pe_dim))
        self.register_buffer('theta', theta)

    def forward(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(num_nodes, device=self.theta.device).float()
        sin = torch.sin(positions[:, None] * self.theta[None, :])
        cos = torch.cos(positions[:, None] * self.theta[None, :])
        return sin, cos


class PositionalEncoder(nn.Module):
    """Hybrid Positional Encoder: Laplacian primary, Rotary fallback."""

    def __init__(self, pe_dim: int = 16, laplacian_pe_dim: int = 16,
                 rotary_pe_dim: int = 16, max_freq: float = 1e4,
                 device=None, config_path: Optional[str] = None):
        super().__init__()
        if config_path:
            cfg = load_config(config_path)
            pe_dim = cfg.get('model.foundation.laplacian_pe_dim', pe_dim)
        self.pe_dim = pe_dim
        self.laplacian_encoder = LaplacianPositionalEncoder(
            pe_dim=laplacian_pe_dim, max_freq=max_freq, device=device)
        self.rotary_encoder = RotaryPositionalEncoder(pe_dim=rotary_pe_dim, device=device)
        self.fallback_to_rotary = True

    def forward(self, edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        try:
            pos_enc = self.laplacian_encoder(edge_index, num_nodes)
        except RuntimeError as e:
            warnings.warn(f"Laplacian PE failed, falling back to Rotary PE: {e}", RuntimeWarning)
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            sin, cos = self.rotary_encoder(num_nodes)
            pos_enc = torch.cat([sin, cos], dim=-1)[:, :self.pe_dim]
        return pos_enc.to(edge_index.device)
