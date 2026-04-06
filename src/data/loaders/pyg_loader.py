"""
PyTorch Geometric dataset loader with retry logic and synthetic fallback.

Loads datasets from PyG (Cora, PubMed, CiteSeer, Amazon) with automatic
retry and fallback to synthetic graphs on persistent failures.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch_geometric.transforms as T
from loguru import logger
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    CoraFull,
    CitationFull,
)

from src.data.graph_data import GraphData
from src.data.loaders.base_loader import BaseGraphLoader
from src.data.loaders.synthetic_loader import SyntheticGraphLoader


class PyGDatasetLoader(BaseGraphLoader):
    """
    Load PyTorch Geometric datasets with retry and fallback.

    Supports: Cora, PubMed, CiteSeer, Amazon Computers, Amazon Photo
    """

    # Map dataset names to PyG dataset classes
    DATASET_REGISTRY = {
        "cora": (Planetoid, {"name": "Cora"}),
        "pubmed": (Planetoid, {"name": "PubMed"}),
        "citeseer": (Planetoid, {"name": "CiteSeer"}),
        "amazon_computers": (Amazon, {"name": "Computers"}),
        "amazon_photo": (Amazon, {"name": "Photo"}),
        "karate_club": ("karate_club", {}),  # Special handling
    }

    def __init__(
        self,
        name: str,
        cache_dir: str = "data/cache",
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        use_synthetic_fallback: bool = True,
    ):
        """
        Initialize PyG dataset loader.

        Args:
            name: Dataset name (see DATASET_REGISTRY keys)
            cache_dir: Cache directory for downloads
            max_retries: Max retry attempts on download failure
            retry_delay_seconds: Delay between retries
            use_synthetic_fallback: Use synthetic graph if real dataset fails

        Raises:
            ValueError: If dataset name not in registry
        """
        if name not in self.DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(self.DATASET_REGISTRY.keys())}"
            )

        super().__init__(name, cache_dir)

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.use_synthetic_fallback = use_synthetic_fallback
        self.cache_dir = Path(cache_dir)

    def load(self) -> GraphData:
        """
        Load dataset with retry logic and fallback.

        Returns:
            GraphData: Loaded graph

        Raises:
            RuntimeError: If loading fails and synthetic fallback disabled
        """
        # Try real dataset
        try:
            return self._load_with_retries()
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.name}': {e}")

            if self.use_synthetic_fallback:
                logger.warning(f"Falling back to synthetic graph for '{self.name}'")
                return self._load_synthetic_fallback()
            else:
                raise RuntimeError(f"Failed to load {self.name} and synthetic fallback disabled") from e

    def _load_with_retries(self) -> GraphData:
        """
        Load dataset with automatic retries.

        Returns:
            GraphData: Loaded graph

        Raises:
            Exception: If all retries exhausted
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Loading {self.name} (attempt {attempt}/{self.max_retries})")
                return self._load_raw()
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay_seconds} seconds...")
                    time.sleep(self.retry_delay_seconds)

        raise last_error

    def _load_raw(self) -> GraphData:
        """
        Load raw dataset from PyG.

        Returns:
            GraphData: Loaded graph

        Raises:
            RuntimeError: If dataset loading fails
        """
        if self.name == "karate_club":
            return self._load_karate_club()

        dataset_class, kwargs = self.DATASET_REGISTRY[self.name]

        try:
            # Create dataset
            dataset = dataset_class(
                root=str(self.cache_dir),
                transform=T.NormalizeFeatures(),
                **kwargs,
            )

            # Get first graph
            data = dataset[0]

            logger.info(
                f"Loaded {self.name}: {data.num_nodes} nodes, {data.num_edges} edges"
            )

            return GraphData(
                node_features=data.x,
                edge_index=data.edge_index,
                edge_labels=None,  # Will be added during negative sampling
                graph_metadata={
                    "domain": self.name,
                    "source": "pytorch_geometric",
                    "num_classes": dataset.num_classes,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load {self.name} from PyG: {e}") from e

    def _load_karate_club(self) -> GraphData:
        """
        Load Zachary's Karate Club dataset.

        Returns:
            GraphData: Loaded graph

        Raises:
            RuntimeError: If loading fails
        """
        try:
            from torch_geometric.datasets import KarateClub

            # KarateClub does not accept 'root' in newer PyG versions
            try:
                dataset = KarateClub()
            except TypeError:
                dataset = KarateClub(root=str(self.cache_dir))
            data = dataset[0]

            logger.info(
                f"Loaded karate_club: {data.num_nodes} nodes, {data.num_edges} edges"
            )

            return GraphData(
                node_features=data.x if data.x is not None else torch.eye(data.num_nodes),
                edge_index=data.edge_index,
                edge_labels=None,
                graph_metadata={
                    "domain": "karate_club",
                    "source": "pytorch_geometric",
                    "num_classes": 2,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load karate_club: {e}") from e

    def _load_synthetic_fallback(self) -> GraphData:
        """
        Create synthetic fallback graph.

        Returns:
            GraphData: Generated synthetic graph
        """
        # Estimate reasonable graph size from dataset name
        num_nodes = 100
        num_features = 64

        if "amazon" in self.name:
            num_nodes = 250
            num_features = 100
        elif "pubmed" in self.name:
            num_nodes = 200
            num_features = 500
        elif "cora_full" in self.name:
            num_nodes = 300

        synthetic_loader = SyntheticGraphLoader(
            num_nodes=num_nodes,
            num_features=num_features,
            num_edges=None,  # Will auto-compute
            avg_degree=5.0,
            seed=42,
        )

        graph = synthetic_loader.load()
        graph.graph_metadata["domain"] = self.name
        graph.graph_metadata["source"] = "synthetic_fallback"

        logger.info(f"Created synthetic fallback: {graph.summary()}")
        return graph
