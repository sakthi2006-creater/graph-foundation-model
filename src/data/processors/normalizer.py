"""
Feature normalization utilities.

Normalization transformations: zero-mean, unit-variance, and standardization.
"""

from typing import Optional, Tuple

import torch
from loguru import logger

from src.data.graph_data import GraphData


class FeatureNormalizer:
    """Normalize node features to zero mean and unit variance."""

    def __init__(self, eps: float = 1e-8):
        """
        Initialize normalizer.

        Args:
            eps: Small constant to avoid division by zero
        """
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, graph: GraphData) -> "FeatureNormalizer":
        """
        Compute normalization statistics from graph.

        Args:
            graph: Graph to fit normalization on

        Returns:
            self for chaining
        """
        self.mean = graph.node_features.mean(dim=0)
        self.std = graph.node_features.std(dim=0)

        logger.info(
            f"Fitted normalizer: mean={self.mean.mean():.4f}, std={self.std.mean():.4f}"
        )

        return self

    def transform(self, graph: GraphData) -> GraphData:
        """
        Apply normalization to graph.

        Args:
            graph: Graph to normalize

        Returns:
            GraphData with normalized features

        Raises:
            RuntimeError: If normalizer not fitted
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        # Normalize
        normalized_features = (graph.node_features - self.mean) / (self.std + self.eps)

        return GraphData(
            node_features=normalized_features,
            edge_index=graph.edge_index,
            edge_labels=graph.edge_labels,
            graph_metadata=graph.graph_metadata.copy(),
        )

    def fit_transform(self, graph: GraphData) -> GraphData:
        """
        Fit and transform in one step.

        Args:
            graph: Graph to normalize

        Returns:
            GraphData with normalized features
        """
        return self.fit(graph).transform(graph)

    def inverse_transform(self, normalized_features: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization (denormalize).

        Args:
            normalized_features: Normalized features

        Returns:
            Original scale features

        Raises:
            RuntimeError: If normalizer not fitted
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted.")

        return normalized_features * (self.std + self.eps) + self.mean
