"""
Negative sampling utilities.

Generate negative edges (non-existent edges) for link prediction training.
"""

from typing import Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger

from src.data.graph_data import GraphData


class NegativeSampler:
    """Generate negative edge samples (non-existent edges)."""

    def __init__(
        self,
        strategy: str = "degree_weighted",
        num_negative_per_positive: int = 1,
        seed: int = 42,
    ):
        """
        Initialize negative sampler.

        Args:
            strategy: Sampling strategy ('degree_weighted', 'uniform', 'random_walk')
            num_negative_per_positive: Ratio of negative to positive edges
            seed: Random seed

        Raises:
            ValueError: If strategy unknown
        """
        if strategy not in ("degree_weighted", "uniform", "random_walk"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.num_negative_per_positive = num_negative_per_positive
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample(self, graph: GraphData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative edge samples.

        Returns:
            Tuple of (negative_edge_index, labels)
            where labels are all zeros (negative label)

        Raises:
            ValueError: If cannot generate unique negative samples
        """
        if graph.num_edges == 0:
            raise ValueError("Cannot sample from graph with zero edges")

        num_negative = graph.num_edges * self.num_negative_per_positive

        if self.strategy == "degree_weighted":
            neg_edges = self._sample_degree_weighted(graph, num_negative)
        elif self.strategy == "uniform":
            neg_edges = self._sample_uniform(graph, num_negative)
        else:  # random_walk
            neg_edges = self._sample_random_walk(graph, num_negative)

        # Create edge index and labels
        negative_edge_index = torch.tensor(
            neg_edges, dtype=torch.long
        ).t().contiguous()
        negative_labels = torch.zeros(negative_edge_index.shape[1], dtype=torch.long)

        logger.info(f"Sampled {negative_edge_index.shape[1]} negative edges ({self.strategy})")

        return negative_edge_index, negative_labels

    def _sample_degree_weighted(
        self, graph: GraphData, num_samples: int
    ) -> list:
        """
        Sample negative edges weighted by node degree.

        Args:
            graph: Graph to sample from
            num_samples: Number of negative samples

        Returns:
            List of edge tuples
        """
        # Get existing edges as set
        edge_set = set()
        for i in range(graph.num_edges):
            u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            edge_set.add((min(u, v), max(u, v)))  # Undirected
            edge_set.add((max(u, v), min(u, v)))  # Both directions for undirected

        # Compute node degrees
        degrees = torch.zeros(graph.num_nodes, dtype=torch.float)
        for i in range(graph.num_edges):
            u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            degrees[u] += 1
            degrees[v] += 1

        # Normalize to probability
        degree_probs = degrees / (degrees.sum() + 1e-8)

        # Sample negative edges
        negative_edges = []
        max_attempts = num_samples * 10

        for _ in range(max_attempts):
            if len(negative_edges) >= num_samples:
                break

            # Sample two nodes by degree
            u = self.rng.choice(graph.num_nodes, p=degree_probs.numpy())
            v = self.rng.choice(graph.num_nodes, p=degree_probs.numpy())

            # Avoid self-loops and existing edges
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in edge_set:
                    negative_edges.append([u, v])
                    edge_set.add(edge)
                    edge_set.add((v, u))

        if len(negative_edges) < num_samples:
            logger.warning(f"Requested {num_samples} negatives, got {len(negative_edges)}")

        return negative_edges

    def _sample_uniform(self, graph: GraphData, num_samples: int) -> list:
        """
        Sample negative edges uniformly at random.

        Args:
            graph: Graph to sample from
            num_samples: Number of negative samples

        Returns:
            List of edge tuples
        """
        # Get existing edges
        edge_set = set()
        for i in range(graph.num_edges):
            u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            edge_set.add((min(u, v), max(u, v)))

        negative_edges = []
        max_attempts = num_samples * 10

        for _ in range(max_attempts):
            if len(negative_edges) >= num_samples:
                break

            u = self.rng.randint(0, graph.num_nodes)
            v = self.rng.randint(0, graph.num_nodes)

            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in edge_set:
                    negative_edges.append([u, v])
                    edge_set.add(edge)

        return negative_edges

    def _sample_random_walk(self, graph: GraphData, num_samples: int) -> list:
        """
        Sample negative edges via random walk (more realistic).

        Args:
            graph: Graph to sample from
            num_samples: Number of negative samples

        Returns:
            List of edge tuples
        """
        # Build adjacency dict
        adj = {}
        for i in range(graph.num_nodes):
            adj[i] = []

        for i in range(graph.num_edges):
            u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            adj[u].append(v)
            adj[v].append(u)

        edge_set = set()
        for i in range(graph.num_edges):
            u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            edge_set.add((min(u, v), max(u, v)))

        negative_edges = []
        max_attempts = num_samples * 10

        for _ in range(max_attempts):
            if len(negative_edges) >= num_samples:
                break

            # Start from random node
            u = self.rng.randint(0, graph.num_nodes)

            # Walk 2 steps
            if adj[u]:
                v = self.rng.choice(adj[u])
                if adj[v]:
                    w = self.rng.choice(adj[v])

                    if u != w:
                        edge = (min(u, w), max(u, w))
                        if edge not in edge_set:
                            negative_edges.append([u, w])
                            edge_set.add(edge)

        return negative_edges
