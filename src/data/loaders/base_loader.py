"""
Abstract base class for graph dataset loaders.

Defines LoaderBase interface that all dataset loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from loguru import logger

from src.data.graph_data import GraphData


class BaseGraphLoader(ABC):
    """
    Abstract base class for graph dataset loaders.

    All subclasses must implement load() method to return GraphData objects.
    """

    def __init__(self, name: str, cache_dir: Optional[str] = None):
        """
        Initialize base loader.

        Args:
            name: Dataset name (e.g., 'cora', 'pubmed')
            cache_dir: Optional directory to cache downloaded datasets
        """
        self.name = name
        self.cache_dir = cache_dir
        logger.info(f"Initialized {self.__class__.__name__} for dataset '{name}'")

    @abstractmethod
    def load(self) -> GraphData:
        """
        Load and return graph data.

        Must be implemented by subclasses.

        Returns:
            GraphData: Loaded graph with node_features, edge_index, metadata

        Raises:
            RuntimeError: If data loading fails
            FileNotFoundError: If dataset not found
        """
        pass

    def download(self) -> None:
        """
        Download dataset if not already cached.

        Override in subclasses if download is needed.
        """
        pass

    def validate(self, graph: GraphData) -> bool:
        """
        Validate loaded graph structure.

        Args:
            graph: GraphData to validate

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        if graph.num_nodes <= 0:
            raise ValueError(f"Graph has no nodes: {graph.num_nodes}")

        if graph.num_edges < 0:
            raise ValueError(f"Graph has negative edges: {graph.num_edges}")

        if graph.num_features <= 0:
            raise ValueError(f"Graph has no features: {graph.num_features}")

        logger.info(f"Graph validation passed: {graph.summary()}")
        return True

    def __call__(self) -> GraphData:
        """Calling loader instance loads the data."""
        return self.load()
