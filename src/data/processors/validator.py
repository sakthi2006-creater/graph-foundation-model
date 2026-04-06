"""
Graph validation utilities.

Validate graph structure, detect anomalies, and report statistics.
"""

from typing import Dict, List, Optional

import torch
from loguru import logger

from src.data.graph_data import GraphData


class GraphValidator:
    """Validate graph structure and integrity."""

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, raise exceptions on validation failures
                   If False, only warn
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, graph: GraphData) -> bool:
        """
        Perform comprehensive validation.

        Args:
            graph: GraphData to validate

        Returns:
            bool: True if all checks pass

        Raises:
            ValueError: If strict=True and validation fails
        """
        self.errors.clear()
        self.warnings.clear()

        self._check_tensors(graph)
        self._check_edge_index(graph)
        self._check_features(graph)
        self._check_anomalies(graph)

        if self.errors:
            error_msg = "Graph validation FAILED:\n" + "\n".join(f"  [ERROR] {e}" for e in self.errors)
            logger.error(error_msg)

            if self.strict:
                raise ValueError(error_msg)
            return False

        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  [WARN] {w}")

        logger.info(f"Graph validation PASSED: {graph.summary()}")
        return True

    def _check_tensors(self, graph: GraphData) -> None:
        """Check that tensors are valid PyTorch tensors."""
        if not isinstance(graph.node_features, torch.Tensor):
            self.errors.append(
                f"node_features is not torch.Tensor: {type(graph.node_features)}"
            )

        if not isinstance(graph.edge_index, torch.Tensor):
            self.errors.append(
                f"edge_index is not torch.Tensor: {type(graph.edge_index)}"
            )

        if graph.edge_labels is not None and not isinstance(graph.edge_labels, torch.Tensor):
            self.errors.append(
                f"edge_labels is not torch.Tensor: {type(graph.edge_labels)}"
            )

    def _check_edge_index(self, graph: GraphData) -> None:
        """Check edge_index validity."""
        if graph.edge_index.shape[0] != 2:
            self.errors.append(
                f"edge_index shape[0] must be 2, got {graph.edge_index.shape[0]}"
            )
            return

        if graph.edge_index.dtype not in (torch.int32, torch.int64):
            self.warnings.append(
                f"edge_index dtype should be int32 or int64, got {graph.edge_index.dtype}"
            )

        # Check valid node indices
        if graph.num_edges > 0:
            max_idx = graph.edge_index.max().item()
            min_idx = graph.edge_index.min().item()

            if max_idx >= graph.num_nodes:
                self.errors.append(
                    f"edge_index contains invalid node ID {max_idx} (max valid: {graph.num_nodes - 1})"
                )

            if min_idx < 0:
                self.errors.append(
                    f"edge_index contains negative node ID {min_idx}"
                )

        # Check for self-loops
        if graph.num_edges > 0:
            self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
            if self_loops > 0:
                self.warnings.append(f"Graph has {self_loops} self-loops")

    def _check_features(self, graph: GraphData) -> None:
        """Check node features."""
        if graph.node_features.shape[0] != graph.num_nodes:
            self.errors.append(
                f"node_features has {graph.node_features.shape[0]} nodes, "
                f"but num_nodes is {graph.num_nodes}"
            )

        if graph.node_features.dim() != 2:
            self.errors.append(
                f"node_features must be 2D, got shape {graph.node_features.shape}"
            )

        if graph.num_features <= 0:
            self.errors.append(f"num_features must be positive, got {graph.num_features}")

        # Check for NaN/Inf
        if torch.isnan(graph.node_features).any():
            self.errors.append("node_features contains NaN values")

        if torch.isinf(graph.node_features).any():
            self.warnings.append("node_features contains Inf values")

    def _check_anomalies(self, graph: GraphData) -> None:
        """Check for anomalies and unusual patterns."""
        if graph.num_nodes == 0:
            self.errors.append("Graph has zero nodes")

        if graph.num_nodes == 1:
            self.warnings.append("Graph has only 1 node")

        if graph.num_edges == 0:
            self.warnings.append("Graph has zero edges")

        # Check density
        max_edges = graph.num_nodes * (graph.num_nodes - 1) / 2
        density = graph.num_edges / max_edges if max_edges > 0 else 0

        if density > 0.8:
            self.warnings.append(f"Graph is very dense (density={density:.2%})")

        if density < 0.01:
            self.warnings.append(f"Graph is very sparse (density={density:.2%})")

        # Check disconnected issue
        if graph.avg_degree < 1:
            self.warnings.append(
                f"Average degree very low ({graph.avg_degree:.2f}), "
                f"graph likely has disconnected components"
            )

    def get_report(self) -> Dict[str, any]:
        """
        Get detailed validation report.

        Returns:
            Dictionary with validation results
        """
        return {
            "status": "PASSED" if not self.errors else "FAILED",
            "errors": self.errors,
            "warnings": self.warnings,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
        }

    def summary(self) -> str:
        """Get validation summary string."""
        status = "PASSED" if not self.errors else "FAILED"
        return (
            f"{status}\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}"
        )
