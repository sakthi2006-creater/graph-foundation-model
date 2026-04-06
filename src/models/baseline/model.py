"""GraphSAGE Baseline Model for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional

from src.config_loader import load_config
from src.utils import detect_device


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline implementation for comparison."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.3, config_path: str = 'config.yaml', device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = load_config(config_path)
        self.device = device or detect_device()

        conv_dims = self.config.get('model.baseline.graphsage.hidden_dims') or [hidden_dim] * num_layers
        in_dim = self.config.get('data.feature_dim') or hidden_dim

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, conv_dims[0], normalize=True))
        for i in range(1, len(conv_dims)):
            self.convs.append(SAGEConv(conv_dims[i - 1], conv_dims[i], normalize=True))

        self.link_pred = nn.Sequential(
            nn.Linear(2 * conv_dims[-1], hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self._input_dim = in_dim

    def _lazy_init(self, in_dim: int):
        if in_dim != self._input_dim:
            self._input_dim = in_dim
            old_conv = self.convs[0]
            self.convs[0] = SAGEConv(in_dim, old_conv.out_channels, normalize=True).to(self.device)

    def forward(self, data, edge_index_query=None):
        if hasattr(data, 'node_features'):
            x, edge_index = data.node_features, data.edge_index
        else:
            x, edge_index = data.x, data.edge_index

        x = x.to(self.device).float()
        edge_index = edge_index.to(self.device)
        self._lazy_init(x.shape[-1])

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)

        outputs = {'node_emb': x}
        if edge_index_query is not None:
            row, col = edge_index_query.to(self.device)
            edge_emb = torch.cat([x[row], x[col]], dim=1)
            outputs['link_scores'] = torch.sigmoid(self.link_pred(edge_emb).squeeze())
        return outputs

    def predict_links(self, data, edge_pos, edge_neg):
        node_emb = self(data)['node_emb']
        pos_scores = torch.sigmoid(self.link_pred(
            torch.cat([node_emb[edge_pos[0]], node_emb[edge_pos[1]]], dim=1)).squeeze())
        neg_scores = torch.sigmoid(self.link_pred(
            torch.cat([node_emb[edge_neg[0]], node_emb[edge_neg[1]]], dim=1)).squeeze())
        return pos_scores, neg_scores
