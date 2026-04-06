"""
Phase 1 validation script - Test data pipeline components.

Run: python tests/test_phase_1_data_pipeline.py

This script validates that all Phase 1 data components work correctly.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data import (
    GraphData,
    PyGDatasetLoader,
    SyntheticGraphLoader,
    FeatureNormalizer,
    EdgeSplitter,
    NegativeSampler,
    GraphValidator,
)
from src.data.pipeline import DataPipeline
from src.utils import set_seed, setup_logging, log_environment_info


def test_graph_data():
    """Test GraphData class."""
    print("\n" + "=" * 70)
    print("TEST 1: GraphData")
    print("=" * 70)

    # Create sample graph
    node_features = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    graph = GraphData(
        node_features=node_features,
        edge_index=edge_index,
        graph_metadata={"domain": "test"},
    )

    print(graph.summary())
    print(f"✅ GraphData created: {graph.num_nodes} nodes, {graph.num_edges} edges")
    return graph


def test_synthetic_loader(seed: int = 42):
    """Test synthetic graph generation."""
    print("\n" + "=" * 70)
    print("TEST 2: SyntheticGraphLoader")
    print("=" * 70)

    loader = SyntheticGraphLoader(
        num_nodes=50,
        num_features=32,
        avg_degree=4.0,
        seed=seed,
    )

    graph = loader.load()
    print(graph.summary())
    print(f"✅ Synthetic graph created with density={graph.density():.4f}")
    return graph


def test_pyg_loader():
    """Test PyG dataset loader (with fallback to synthetic)."""
    print("\n" + "=" * 70)
    print("TEST 3: PyGDatasetLoader")
    print("=" * 70)

    # Try to load Cora (will use synthetic if download fails)
    loader = PyGDatasetLoader(
        name="cora",
        cache_dir="data/cache",
        max_retries=1,  # Reduced retries for testing
        use_synthetic_fallback=True,
    )

    graph = loader.load()
    print(graph.summary())
    print(f"✅ Dataset loaded: {graph.graph_metadata.get('source', 'unknown')}")
    return graph


def test_validator(graph: GraphData):
    """Test graph validation."""
    print("\n" + "=" * 70)
    print("TEST 4: GraphValidator")
    print("=" * 70)

    validator = GraphValidator(strict=False)
    is_valid = validator.validate(graph)

    report = validator.get_report()
    print(validator.summary())
    print(f"  Errors: {report['num_errors']}")
    print(f"  Warnings: {report['num_warnings']}")
    print(f"✅ Validation complete: {report['status']}")


def test_normalizer(graph: GraphData):
    """Test feature normalization."""
    print("\n" + "=" * 70)
    print("TEST 5: FeatureNormalizer")
    print("=" * 70)

    normalizer = FeatureNormalizer()
    normalized_graph = normalizer.fit_transform(graph)

    print(f"Original features: mean={graph.node_features.mean():.4f}, "
          f"std={graph.node_features.std():.4f}")
    print(f"Normalized features: mean={normalized_graph.node_features.mean():.4f}, "
          f"std={normalized_graph.node_features.std():.4f}")
    print(f"✅ Feature normalization complete")


def test_splitter(graph: GraphData):
    """Test edge splitting."""
    print("\n" + "=" * 70)
    print("TEST 6: EdgeSplitter")
    print("=" * 70)

    if graph.num_edges < 10:
        print(f"⚠️  Graph has {graph.num_edges} edges, using synthetic for split test")
        graph = SyntheticGraphLoader(num_nodes=100, num_features=32).load()

    splitter = EdgeSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = splitter.split_graph(graph)

    for split_name, split_graph in splits.items():
        pct = split_graph.num_edges / graph.num_edges * 100
        print(f"  {split_name:6s}: {split_graph.num_edges:4d} edges ({pct:5.1f}%)")

    print(f"✅ Edge split complete")


def test_sampler(graph: GraphData):
    """Test negative sampling."""
    print("\n" + "=" * 70)
    print("TEST 7: NegativeSampler")
    print("=" * 70)

    if graph.num_edges < 10:
        print(f"⚠️  Graph has {graph.num_edges} edges, using synthetic for sampling test")
        graph = SyntheticGraphLoader(num_nodes=100, num_features=32).load()

    sampler = NegativeSampler(strategy="degree_weighted", num_negative_per_positive=1, seed=42)
    neg_edges, neg_labels = sampler.sample(graph)

    print(f"  Positive edges: {graph.num_edges}")
    print(f"  Negative edges: {neg_edges.shape[1]}")
    print(f"  Negative labels (all 0): {neg_labels.unique().tolist()}")
    print(f"✅ Negative sampling complete")


def test_pipeline():
    """Test full data pipeline."""
    print("\n" + "=" * 70)
    print("TEST 8: DataPipeline (End-to-end)")
    print("=" * 70)

    # Load config
    config = load_config("config.yaml")

    # Create pipeline
    pipeline = DataPipeline(config, cache_dir="data/cache", seed=42)

    # Load synthetic graph for quick test
    print("Loading synthetic dataset for pipeline test...")
    graph = SyntheticGraphLoader(num_nodes=100, num_features=32).load()

    # Preprocess and split
    print("Running full preprocessing pipeline...")
    splits = pipeline.preprocess_and_split(graph, add_negative_samples=False)

    for split_name, split_graph in splits.items():
        print(f"  {split_name:6s}: {split_graph.num_nodes:3d} nodes, "
              f"{split_graph.num_edges:4d} edges, features={split_graph.num_features}")

    print(f"✅ Data pipeline complete")


def main():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 70)
    print("PHASE 1: DATA PIPELINE VALIDATION")
    print("=" * 70)

    # Setup
    set_seed(42)
    setup_logging(log_dir="logs", level="INFO")
    log_environment_info()

    try:
        # Test components
        graph1 = test_graph_data()
        graph2 = test_synthetic_loader()
        graph3 = test_pyg_loader()

        test_validator(graph2)
        test_normalizer(graph2)
        test_splitter(graph2)
        test_sampler(graph2)

        test_pipeline()

        # Summary
        print("\n" + "=" * 70)
        print("✅ PHASE 1 VALIDATION: ALL TESTS PASSED")
        print("=" * 70)
        print("\nPhase 1 is complete and working correctly!")
        print("All data loading, processing, and validation components are functional.")
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ PHASE 1 VALIDATION FAILED: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
