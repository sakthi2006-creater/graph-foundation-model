"""Multi-Head Graph Attention Mechanism for Graph Transformers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
from src.utils import detect_device


class MultiHeadAttention(MessagePassing):
    """Multi-head attention layer for graph-structured data."""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8,
                 concat: bool = True, dropout: float = 0.1, edge_dropout: float = 0.1,
                 temperature: float = 0.5, bias: bool = True, root_weight: bool = True,
                 device=None, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.dropout_p = dropout
        self.edge_dropout = edge_dropout
        self.temperature = temperature
        self.root_weight = root_weight
        self.head_dim = out_channels // num_heads

        self.lin_query = nn.Linear(in_channels, self.head_dim * num_heads, bias=bias)
        self.lin_key   = nn.Linear(in_channels, self.head_dim * num_heads, bias=bias)
        self.lin_value = nn.Linear(in_channels, self.head_dim * num_heads, bias=bias)

        if root_weight:
            self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_key.weight,   gain=1.414)
        nn.init.xavier_uniform_(self.lin_value.weight, gain=1.414)
        if self.root_weight:
            nn.init.xavier_uniform_(self.lin_root.weight, gain=1.414)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, C = self.num_heads, self.head_dim

        # [N, H, C]
        query = self.lin_query(x).view(-1, H, C)
        key   = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate — passes query_i, key_j, value_j to message()
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             size=(x.size(0), x.size(0)))
        # out: [N, H, C]
        out = out.view(-1, self.out_channels)   # [N, H*C]

        if self.root_weight:
            out = out + self.lin_root(x)

        return out

    def message(self, query_i: torch.Tensor, key_j: torch.Tensor,
                value_j: torch.Tensor, index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor:
        # query_i, key_j: [E, H, C]
        alpha = (query_i * key_j).sum(dim=-1) / self.temperature   # [E, H]
        alpha = softmax(alpha, index, ptr, size_i)                  # [E, H]

        if self.training and self.edge_dropout > 0:
            alpha = F.dropout(alpha, p=self.edge_dropout, training=True)

        return value_j * alpha.unsqueeze(-1)   # [E, H, C]

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out   # [N, H, C]
