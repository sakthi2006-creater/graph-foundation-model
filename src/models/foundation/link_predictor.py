"""Link Prediction Head: MLP for edge score prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.utils import detect_device

class LinkPredictor(nn.Module):
    """MLP Link Predictor Head.
    
    Takes node embeddings u, v -> concat -> MLP -> sigmoid score.
    
    Args:
        hidden_dim: Input embedding dimension
        mlp_dim: Hidden MLP dimension (256 default)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu')
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = nn.Dropout(dropout)
        self.device = device or detect_device()
        
        act_fn = {'relu': nn.ReLU, 'gelu': nn.GELU}[activation]()
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_dim),
            act_fn,
            self.dropout,
            nn.Linear(mlp_dim, mlp_dim // 2),
            act_fn,
            self.dropout,
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def forward(
        self,
        z: torch.Tensor,  # Node embeddings [num_nodes, hidden_dim]
        edge_index: torch.Tensor,  # [2, num_queries]
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """"Predict link probabilities for query edges.
        
        Args:
            z: Node embeddings
            edge_index: Query edge indices [2, num_queries]
            edge_attr: Optional edge features
        
        Returns:
            scores: [num_queries] sigmoid scores [0,1]
        """
        row, col = edge_index
        edge_embeddings = torch.cat([z[row], z[col]], dim=-1)
        
        if edge_attr is not None:
            edge_embeddings = torch.cat([edge_embeddings, edge_attr], dim=-1)
        
        scores = self.mlp(edge_embeddings).squeeze(-1)
        return torch.sigmoid(scores)
    
    def score_pairs(self, emb_u: torch.Tensor, emb_v: torch.Tensor) -> torch.Tensor:
        """Score pairs of embeddings directly.
        
        Args:
            emb_u, emb_v: [batch_size, hidden_dim]
        
        Returns:
            scores: [batch_size]
        """
        concat = torch.cat([emb_u, emb_v], dim=-1)
        scores = self.mlp(concat).squeeze(-1)
        return torch.sigmoid(scores)


if __name__ == "__main__":
    # Test
    device = detect_device()
    hidden_dim = 128
    z = torch.randn(100, hidden_dim, device=device)
    edge_index = torch.randint(0, 100, (2, 32), device=device)
    
    predictor = LinkPredictor(hidden_dim, device=device)
    scores = predictor(z, edge_index)
    
    print(f"Link predictor input edges: {edge_index.shape[1]}")
    print(f"Scores shape: {scores.shape}, range [{scores.min():.3f}, {scores.max():.3f}]")

