"""Graph Transformer Layer with Residual Gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.foundation.attention import MultiHeadAttention
from src.utils import detect_device

class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer with multi-head attention + residual gating.
    
    Architecture per config:
    - MultiHeadAttention
    - Feed-forward (optional)
    - Residual gating: h = h + sigmoid(gate) * ff(h)
    - Layer normalization
    - Dropout
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Attention heads
        dropout: Dropout rate
        edge_dropout: Edge dropout
        temperature: Attention temperature
        residual_gating: Use sigmoid gating
        ff_dim: Feed-forward hidden dim (4*hidden_dim default)
        device: Compute device
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dropout: float = 0.1,
        temperature: float = 0.5,
        residual_gating: bool = True,
        ff_dim: Optional[int] = None,
        device: str = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual_gating = residual_gating
        self.device = device or detect_device()
        
        ff_dim = ff_dim or 4 * hidden_dim
        
        # Sub-layers
        self.attention = MultiHeadAttention(
            hidden_dim, hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            temperature=temperature,
            concat=True,
            device=self.device
        )
        
        self.ff_linear1 = nn.Linear(hidden_dim, ff_dim)
        self.ff_linear2 = nn.Linear(ff_dim, hidden_dim)
        self.ff_dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gating network for residual
        if residual_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        self.attention.reset_parameters()
        nn.init.xavier_uniform_(self.ff_linear1.weight)
        nn.init.xavier_uniform_(self.ff_linear2.weight)
        nn.init.zeros_(self.ff_linear2.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """"Forward pass through transformer layer.
        
        Args:
            x: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: Optional [num_edges, hidden_dim]
        
        Returns:
            out: [num_nodes, hidden_dim]
        """
        # Attention + residual + norm (pre-norm style)
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, edge_index, edge_attr)
        x = self.dropout(attn_out) + residual
        
        # Feed-forward + gating + norm
        residual = x
        x = self.norm2(x)
        ff_out = self.ff_linear2(self.ff_dropout(F.gelu(self.ff_linear1(x))))
        
        if self.residual_gating:
            gate = self.gate(x)
            x = residual + gate * ff_out
        else:
            x = self.dropout(ff_out) + residual
        
        return x

class GraphTransformerBlock(nn.Module):
    """Stack of GraphTransformerLayers."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                **kwargs
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


if __name__ == "__main__":
    # Test
    device = detect_device()
    x = torch.randn(100, 128, device=device, requires_grad=True)
    edge_index = torch.randint(0, 100, (2, 500), device=device)
    edge_attr = torch.randn(500, 128, device=device)
    
    layer = GraphTransformerLayer(128, num_heads=8, device=device)
    out = layer(x, edge_index, edge_attr)
    
    print(f"Transformer layer input: {x.shape}")
    print(f"Transformer layer output: {out.shape}")
    print("Loss for backward:", out.sum())

