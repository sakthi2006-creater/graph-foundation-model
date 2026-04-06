"""
Edge splitting utilities.

Split edges into train/val/test sets with various strategies.
"""

from typing import Dict, Optional, Tuple

import torch
from loguru import logger

from src.data.graph_data import GraphData


class EdgeSplitter:
    """Split graph edges into train, validation, and test sets."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize edge splitter.

        Args:
            train_ratio: Fraction of edges for training
            val_ratio: Fraction of edges for validation
            test_ratio: Fraction of edges for testing
            seed: Random seed for reproducible splits

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        total_ratio = train_ratio + val_ratio + test_ratio

        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self, graph: GraphData
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Split edges into train/val/test.

        Args:
            graph: Graph with edges to split

        Returns:
            Dictionary with keys 'train', 'val', 'test'
            Each value is tuple of (edge_index, edge_labels)

        Raises:
            ValueError: If graph has no edges
        """
        if graph.num_edges == 0:
            raise ValueError("Cannot split graph with zero edges")

        num_edges = graph.num_edges

        # Random permutation
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        perm = torch.randperm(num_edges, generator=generator)

        # Compute split indices
        train_size = int(num_edges * self.train_ratio)
        val_size = int(num_edges * self.val_ratio)

        train_idx = perm[:train_size]
        val_idx = perm[train_size : train_size + val_size]
        test_idx = perm[train_size + val_size :]

        splits = {}

        if graph.edge_labels is not None:
            splits["train"] = (graph.edge_index[:, train_idx], graph.edge_labels[train_idx])
            splits["val"] = (graph.edge_index[:, val_idx], graph.edge_labels[val_idx])
            splits["test"] = (graph.edge_index[:, test_idx], graph.edge_labels[test_idx])
        else:
            splits["train"] = (graph.edge_index[:, train_idx], None)
            splits["val"] = (graph.edge_index[:, val_idx], None)
            splits["test"] = (graph.edge_index[:, test_idx], None)

        logger.info(
            f"Edge split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

        return splits

    def split_graph(self, graph: GraphData) -> Dict[str, GraphData]:
        """
        Split graph into train/val/test subgraphs.

        Note: This creates separate GraphData objects for each split.

        Args:
            graph: Graph to split

        Returns:
            Dictionary with 'train', 'val', 'test' GraphData objects
        """
        splits = self.split(graph)
        result = {}

        for split_name, (edge_index, edge_labels) in splits.items():
            result[split_name] = GraphData(
                node_features=graph.node_features,
                edge_index=edge_index,
                edge_labels=edge_labels,
                graph_metadata=graph.graph_metadata.copy(),
            )

        return result
