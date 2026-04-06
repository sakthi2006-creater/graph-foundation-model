"""Full Graph Foundation Model: GraphTransformer + Link Predictor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.config_loader import load_config
from src.data.graph_data import GraphData
from src.models.positional_encoding import PositionalEncoder
from src.models.foundation.transformer import GraphTransformerBlock
from src.models.foundation.link_predictor import LinkPredictor
from src.utils import detect_device, set_seed


class GraphFoundationModel(nn.Module):
    """Complete Graph Foundation Model.

    Pipeline:
    1. Input projection
    2. Positional encoding (Laplacian/Rotary)
    3. GraphTransformerBlock (4 layers)
    4. LinkPredictor MLP
    """

    def __init__(self, config_path: str = 'config.yaml', device=None, pooling_strategy: str = 'mean'):
        super().__init__()
        self.config = load_config(config_path)
        self.device = device or detect_device()
        self.pooling_strategy = pooling_strategy

        hidden_dim = self.config.get('model.foundation.hidden_dim', 128)
        num_heads = self.config.get('model.foundation.num_heads', 8)
        num_layers = self.config.get('model.foundation.num_layers', 4)
        dropout = self.config.get('model.foundation.dropout', 0.1)
        edge_dropout = self.config.get('model.foundation.edge_dropout', 0.1)
        temperature = self.config.get('model.foundation.attention_temperature', 0.5)
        residual_gating = self.config.get('model.foundation.residual_gating', True)
        laplacian_pe_dim = self.config.get('model.foundation.laplacian_pe_dim', 16)
        mlp_dim = self.config.get('model.foundation.link_prediction_mlp_dim', 256)
        seed = self.config.get('seed', 42)

        self.hidden_dim = hidden_dim

        # Input projection (accepts any feature dim, lazy via first forward)
        self.input_proj = None
        self._input_dim = None

        # Positional encoder
        self.pos_enc = PositionalEncoder(
            pe_dim=laplacian_pe_dim,
            laplacian_pe_dim=laplacian_pe_dim,
            rotary_pe_dim=laplacian_pe_dim,
            device=self.device,
        )
        # PE projection to hidden_dim
        self.pe_proj = nn.Linear(laplacian_pe_dim, hidden_dim)

        # Transformer stack
        self.transformer = GraphTransformerBlock(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            edge_dropout=edge_dropout,
            temperature=temperature,
            residual_gating=residual_gating,
            device=self.device,
        )

        # Link predictor
        self.link_predictor = LinkPredictor(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            device=self.device,
        )

        self.to(self.device)
        set_seed(seed)

    def _get_input_proj(self, in_dim: int) -> nn.Linear:
        if self._input_dim != in_dim:
            self._input_dim = in_dim
            self.input_proj = nn.Linear(in_dim, self.hidden_dim).to(self.device)
        return self.input_proj

    def forward(self, data, edge_index_query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Support both GraphData and PyG Data objects
        if hasattr(data, 'node_features'):
            x = data.node_features
            edge_index = data.edge_index
            edge_attr = None
        else:
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)

        x = x.to(self.device).float()
        edge_index = edge_index.to(self.device)

        # Input projection
        proj = self._get_input_proj(x.shape[-1])
        x = proj(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Positional encoding
        num_nodes = x.shape[0]
        pos_enc = self.pos_enc(edge_index, num_nodes)
        pos_enc = self.pe_proj(pos_enc)
        x = x + pos_enc

        # Transformer layers
        x = self.transformer(x, edge_index, edge_attr)

        outputs = {'node_emb': x}

        if edge_index_query is not None:
            edge_index_query = edge_index_query.to(self.device)
            scores = self.link_predictor(x, edge_index_query)
            outputs['link_scores'] = scores

        return outputs

    def predict_links(self, data, edge_index_pos: torch.Tensor,
                      edge_index_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self(data)['node_emb']
        pos_scores = self.link_predictor(node_emb, edge_index_pos.to(self.device))
        neg_scores = self.link_predictor(node_emb, edge_index_neg.to(self.device))
        return pos_scores, neg_scores
