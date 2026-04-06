"""
End-to-end data pipeline orchestrator.

Coordinates loading, processing, validation, and splitting of graph data.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.config_loader import ConfigLoader
from src.data.graph_data import GraphData
from src.data.loaders import PyGDatasetLoader
from src.data.processors import (
    EdgeSplitter,
    FeatureNormalizer,
    GraphValidator,
    NegativeSampler,
)


class DataPipeline:
    """Orchestrate complete data pipeline from loading to splitting."""

    def __init__(
        self,
        config: ConfigLoader,
        cache_dir: str = "data/cache",
        seed: int = 42,
    ):
        """
        Initialize data pipeline.

        Args:
            config: ConfigLoader instance
            cache_dir: Cache directory for datasets
            seed: Random seed for reproducibility
        """
        self.config = config
        self.cache_dir = cache_dir
        self.seed = seed

        # Initialize components
        self.validator = GraphValidator(strict=False)
        self.normalizer = FeatureNormalizer()
        self.splitter = EdgeSplitter(
            train_ratio=config.get("data.split.train", 0.7),
            val_ratio=config.get("data.split.val", 0.15),
            test_ratio=config.get("data.split.test", 0.15),
            seed=seed,
        )
        self.sampler = NegativeSampler(
            strategy=config.get("data.negative_sampling.strategy", "degree_weighted"),
            num_negative_per_positive=config.get(
                "data.negative_sampling.num_negative_per_positive", 1
            ),
            seed=seed,
        )

        logger.info("Initialized DataPipeline")

    def load_dataset(self, dataset_name: str) -> GraphData:
        """
        Load single dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'cora', 'pubmed')

        Returns:
            GraphData: Loaded and validated graph

        Raises:
            RuntimeError: If loading fails
        """
        logger.info(f"Loading dataset: {dataset_name}")

        loader = PyGDatasetLoader(
            name=dataset_name,
            cache_dir=self.cache_dir,
            max_retries=self.config.get("data.download_retries", 3),
            retry_delay_seconds=self.config.get("data.retry_delay_seconds", 5.0),
            use_synthetic_fallback=True,
        )

        graph = loader.load()

        # Validate
        self.validator.validate(graph)

        return graph

    def load_all_source_domains(self) -> Dict[str, GraphData]:
        """
        Load all source domain datasets.

        Returns:
            Dictionary mapping domain names to GraphData objects

        Raises:
            RuntimeError: If any dataset fails to load
        """
        source_domains = self.config.get("data.source_domains", [])
        logger.info(f"Loading {len(source_domains)} source domains: {source_domains}")

        graphs = {}
        for domain in source_domains:
            try:
                graphs[domain] = self.load_dataset(domain)
            except Exception as e:
                logger.error(f"Failed to load {domain}: {e}")
                raise RuntimeError(f"Failed to load source domain {domain}") from e

        logger.info(f"Successfully loaded {len(graphs)} source domains")
        return graphs

    def load_target_domain(self) -> GraphData:
        """
        Load target domain dataset.

        Returns:
            GraphData: Target domain graph

        Raises:
            RuntimeError: If loading fails
        """
        target_domain = self.config.get("data.target_domain", "amazon_photo")
        logger.info(f"Loading target domain: {target_domain}")
        return self.load_dataset(target_domain)

    def process_graph(
        self,
        graph: GraphData,
        normalize: bool = True,
        add_negative_samples: bool = False,
    ) -> GraphData:
        """
        Process graph: normalize features and optionally add negative samples.

        Args:
            graph: GraphData to process
            normalize: Whether to normalize node features
            add_negative_samples: Whether to add negative edge labels

        Returns:
            Processed GraphData

        Raises:
            ValueError: If processing fails
        """
        logger.info(f"Processing graph: {graph.summary()}")

        # Normalize features
        if normalize:
            graph = self.normalizer.fit_transform(graph)
            logger.info("Applied feature normalization")

        # Add negative samples
        if add_negative_samples:
            pos_edge_index = graph.edge_index
            pos_labels = graph.edge_labels

            # Generate negative edges
            neg_edge_index, neg_labels = self.sampler.sample(graph)

            # Combine positive and negative
            combined_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            combined_labels = torch.cat([pos_labels or torch.ones(pos_edge_index.shape[1]), neg_labels])

            graph = GraphData(
                node_features=graph.node_features,
                edge_index=combined_edge_index,
                edge_labels=combined_labels,
                graph_metadata=graph.graph_metadata.copy(),
            )

            logger.info(f"Added negative samples: {neg_edge_index.shape[1]} negatives")

        return graph

    def split_edges(self, graph: GraphData) -> Dict[str, Tuple[GraphData, GraphData]]:
        """
        Split graph edges into train/val/test.

        Args:
            graph: GraphData to split

        Returns:
            Dictionary with 'train', 'val', 'test' GraphData objects

        Raises:
            ValueError: If splitting fails
        """
        logger.info(f"Splitting edges for graph with {graph.num_edges} edges")

        splits = self.splitter.split_graph(graph)

        for split_name, split_graph in splits.items():
            logger.info(
                f"  {split_name}: {split_graph.num_edges} edges "
                f"({split_graph.num_edges / graph.num_edges * 100:.1f}%)"
            )

        return splits

    def preprocess_and_split(
        self,
        graph: GraphData,
        add_negative_samples: bool = False,
    ) -> Dict[str, GraphData]:
        """
        Full preprocessing: normalize → add negatives → split edges.

        Args:
            graph: GraphData to process
            add_negative_samples: Whether to add negative edges

        Returns:
            Dictionary with 'train', 'val', 'test' processed GraphData objects

        Raises:
            ValueError: If processing fails
        """
        logger.info("Starting full preprocessing pipeline")

        # Process
        graph = self.process_graph(
            graph,
            normalize=True,
            add_negative_samples=add_negative_samples,
        )

        # Split
        splits = self.split_edges(graph)

        logger.info("Preprocessing pipeline complete")
        return splits


# Import torch at end to avoid circular imports
import torch
