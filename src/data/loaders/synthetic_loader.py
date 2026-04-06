"""
Synthetic graph generator for fallback and testing.

Creates random graphs with configurable parameters for testing and as fallback
when real dataset downloads fail.
"""

from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.data.graph_data import GraphData
from src.data.loaders.base_loader import BaseGraphLoader


class SyntheticGraphLoader(BaseGraphLoader):
    """Generate synthetic random graphs for testing and fallback."""

    def __init__(
        self,
        num_nodes: int = 100,
        num_features: int = 64,
        num_edges: Optional[int] = None,
        avg_degree: float = 5.0,
        seed: int = 42,
    ):
        """
        Initialize synthetic graph generator.

        Args:
            num_nodes: Number of nodes
            num_features: Number of node features
            num_edges: Number of edges (if None, computed from avg_degree)
            avg_degree: Average node degree (used if num_edges is None)
            seed: Random seed

        Raises:
            ValueError: If parameters invalid
        """
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if avg_degree <= 0:
            raise ValueError(f"avg_degree must be positive, got {avg_degree}")

        super().__init__("synthetic", cache_dir=None)

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.avg_degree = avg_degree
        self.seed = seed

        # Compute num_edges from avg_degree if not specified
        if num_edges is None:
            self.num_edges = max(1, int(num_nodes * avg_degree / 2))
        else:
            if num_edges <= 0:
                raise ValueError(f"num_edges must be positive, got {num_edges}")
            self.num_edges = num_edges

    def load(self) -> GraphData:
        """
        Generate synthetic graph.

        Returns:
            GraphData: Generated random graph

        Raises:
            RuntimeError: If generation fails
        """
        try:
            rng = np.random.RandomState(self.seed)

            # Generate random node features
            node_features = torch.randn(self.num_nodes, self.num_features, dtype=torch.float32)

            # Normalize features
            node_features = (node_features - node_features.mean(dim=0)) / (
                node_features.std(dim=0) + 1e-8
            )

            # Generate random edges (without self-loops)
            edges = set()
            max_attempts = self.num_edges * 10

            for _ in range(max_attempts):
                if len(edges) >= self.num_edges:
                    break

                u = rng.randint(0, self.num_nodes)
                v = rng.randint(0, self.num_nodes)

                # Avoid self-loops and duplicates
                if u != v:
                    edge = tuple(sorted([u, v]))  # Undirected
                    edges.add(edge)

            # Convert to edge_index
            if edges:
                edges_list = list(edges)
                edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
            else:
                # Fallback: create at least one edge
                edge_index = torch.tensor(
                    [[0, 1], [1, 0]], dtype=torch.long
                ).t().contiguous()

            logger.info(
                f"Generated synthetic graph: {self.num_nodes} nodes, "
                f"{edge_index.shape[1]} edges, {self.num_features} features"
            )

            return GraphData(
                node_features=node_features,
                edge_index=edge_index,
                edge_labels=None,
                graph_metadata={
                    "domain": "synthetic",
                    "source": "synthetic_generator",
                    "num_classes": 2,
                    "seed": self.seed,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic graph: {e}") from e
