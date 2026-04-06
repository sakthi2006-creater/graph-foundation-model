"""
Graph data structure and validation.

Defines GraphData: a unified representation of attributed graphs with edges,
node features, and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger


@dataclass
class GraphData:
    """
    Unified graph representation with node features, edge index, and metadata.

    Attributes:
        node_features: Node feature matrix [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        edge_labels: Edge labels (1=real, 0=negative) [num_edges]
        graph_metadata: Dictionary with graph statistics and domain info
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_labels: Optional[torch.Tensor] = None
    graph_metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize tensors after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate graph structure and tensor dimensions.

        Raises:
            ValueError: If graph structure is invalid
        """
        # Validate node features
        if not isinstance(self.node_features, torch.Tensor):
            raise ValueError(f"node_features must be torch.Tensor, got {type(self.node_features)}")

        if self.node_features.dim() != 2:
            raise ValueError(
                f"node_features must be 2D [num_nodes, num_features], got shape {self.node_features.shape}"
            )

        num_nodes, num_features = self.node_features.shape

        # Validate edge index
        if not isinstance(self.edge_index, torch.Tensor):
            raise ValueError(f"edge_index must be torch.Tensor, got {type(self.edge_index)}")

        if self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], got {self.edge_index.shape}")

        num_edges = self.edge_index.shape[1]

        # Validate edge indices are in valid range
        if (self.edge_index >= num_nodes).any() or (self.edge_index < 0).any():
            raise ValueError(
                f"edge_index contains invalid node IDs (range [0, {num_nodes})), got max {self.edge_index.max()}"
            )

        # Validate edge labels if present
        if self.edge_labels is not None:
            if not isinstance(self.edge_labels, torch.Tensor):
                raise ValueError(f"edge_labels must be torch.Tensor, got {type(self.edge_labels)}")

            if self.edge_labels.shape[0] != num_edges:
                raise ValueError(
                    f"edge_labels must have length {num_edges}, got {self.edge_labels.shape[0]}"
                )

        # Update metadata
        if not self.graph_metadata:
            self.graph_metadata = {}

        self.graph_metadata.update(
            {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_features": num_features,
                "avg_degree": num_edges / num_nodes if num_nodes > 0 else 0.0,
            }
        )

        logger.debug(
            f"Graph validated: {num_nodes} nodes, {num_edges} edges, {num_features} features"
        )

    def to(self, device: torch.device | str) -> "GraphData":
        """
        Move tensors to device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')

        Returns:
            GraphData with tensors on new device
        """
        graph = GraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_labels=self.edge_labels.to(device) if self.edge_labels is not None else None,
            graph_metadata=self.graph_metadata.copy(),
        )
        return graph

    def cpu(self) -> "GraphData":
        """Move tensors to CPU. Returns: GraphData on CPU."""
        return self.to("cpu")

    def cuda(self, device: int = 0) -> "GraphData":
        """Move tensors to CUDA. Returns: GraphData on CUDA."""
        return self.to(f"cuda:{device}")

    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph."""
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of edges in graph."""
        return self.edge_index.shape[1]

    @property
    def num_features(self) -> int:
        """Number of node features."""
        return self.node_features.shape[1]

    @property
    def device(self) -> torch.device:
        """Device of tensors."""
        return self.node_features.device

    @property
    def avg_degree(self) -> float:
        """Average node degree."""
        return 2 * self.num_edges / self.num_nodes if self.num_nodes > 0 else 0.0

    def density(self) -> float:
        """
        Compute graph density.

        Returns:
            float: Graph density (0 to 1)
        """
        if self.num_nodes <= 1:
            return 0.0
        max_edges = self.num_nodes * (self.num_nodes - 1) / 2
        return self.num_edges / max_edges

    def is_directed(self) -> bool:
        """
        Check if graph is directed (has both (u,v) and (v,u) edges).

        Returns:
            bool: True if directed, False if undirected
        """
        if self.num_edges == 0:
            return False

        # Create set of edges and reverse edges
        edge_set = set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))
        reverse_set = set(zip(self.edge_index[1].tolist(), self.edge_index[0].tolist()))

        # If edge sets intersect, graph has pairs (u,v) and (v,u)
        return bool(edge_set & reverse_set)

    def get_edge_split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Split edges into train/val/test sets.

        Args:
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation (default: 0.15)

        Returns:
            Dictionary with 'train', 'val', 'test' splits
            Each split is tuple of (edge_index, edge_labels)

        Raises:
            ValueError: If ratios don't sum to ~1.0
        """
        test_ratio = 1.0 - train_ratio - val_ratio

        if test_ratio < 0 or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        num_edges = self.num_edges
        train_size = int(num_edges * train_ratio)
        val_size = int(num_edges * val_ratio)

        # Random permutation
        perm = torch.randperm(num_edges)

        train_indices = perm[:train_size]
        val_indices = perm[train_size : train_size + val_size]
        test_indices = perm[train_size + val_size :]

        splits = {}

        if self.edge_labels is not None:
            splits["train"] = (
                self.edge_index[:, train_indices],
                self.edge_labels[train_indices],
            )
            splits["val"] = (
                self.edge_index[:, val_indices],
                self.edge_labels[val_indices],
            )
            splits["test"] = (
                self.edge_index[:, test_indices],
                self.edge_labels[test_indices],
            )
        else:
            splits["train"] = (self.edge_index[:, train_indices], None)
            splits["val"] = (self.edge_index[:, val_indices], None)
            splits["test"] = (self.edge_index[:, test_indices], None)

        return splits

    def summary(self) -> str:
        """
        Get graph summary string.

        Returns:
            str: Human-readable graph statistics
        """
        return (
            f"GraphData(\n"
            f"  nodes={self.num_nodes}, edges={self.num_edges}, features={self.num_features}\n"
            f"  avg_degree={self.avg_degree:.2f}, density={self.density():.4f}\n"
            f"  domain={self.graph_metadata.get('domain', 'unknown')}\n"
            f"  device={self.device}\n"
            f")"
        )

    def __repr__(self) -> str:
        """String representation."""
        return self.summary()
