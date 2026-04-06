"""
Phase 2 validation tests for model architecture.

Comprehensive test suite covering all model components:
- Positional encoding (Laplacian + Rotary PE)
- Attention mechanisms (scaled dot-product, multi-head)
- Graph Transformer layers and encoder
- Link prediction head and scoring
- Complete models (foundation, baseline, with adapters)

Test execution:
    pytest tests/test_phase_2_models.py -v
    pytest tests/test_phase_2_models.py::test_laplacian_pe -v
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from src.models.positional_encoding import (
    PositionalEncoding,
    RotaryPositionalEncoding,
    PositionalEncodingFactory,
)
from src.models.foundation.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
)
from src.models.foundation.transformer import (
    GraphTransformerLayer,
    GraphTransformerEncoder,
)
from src.models.foundation.link_predictor import (
    LinkPredictionHead,
    AdapterModule,
    EdgeScoringHead,
)
from src.models.foundation.model import (
    GraphFoundationEncoder,
    GraphFoundationLinkPredictor,
    GraphFoundationWithAdapter,
)
from src.models.baseline.model import GraphSAGEBaseline, GraphSAGEWithAdapter
from src.models.adapter.model import (
    BottleneckAdapter,
    CompactAdapter,
    LoRAAdapter,
    AdapterModule as AdapterModuleFactory,
)


class TestPositionalEncoding:
    """Test positional encoding modules."""

    def test_laplacian_pe_initialization(self) -> None:
        """Test Laplacian PE initialization and basic properties."""
        pe = PositionalEncoding(hidden_dim=16, num_eig=8)
        assert pe.hidden_dim == 16
        assert pe.num_eig == 8
        assert pe.use_identity is True
        assert hasattr(pe, "proj")

    def test_laplacian_pe_forward(self) -> None:
        """Test Laplacian PE forward pass."""
        pe = PositionalEncoding(hidden_dim=16, num_eig=8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        num_nodes = 4

        output = pe(edge_index, num_nodes)
        assert output.shape == (num_nodes, 16)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_rotary_pe_initialization(self) -> None:
        """Test Rotary PE initialization."""
        pe = RotaryPositionalEncoding(hidden_dim=16, num_nodes=1000)
        assert pe.hidden_dim == 16
        assert pe.num_nodes == 1000
        assert hasattr(pe, "inv_freq")

    def test_rotary_pe_forward(self) -> None:
        """Test Rotary PE forward pass."""
        pe = RotaryPositionalEncoding(hidden_dim=16, num_nodes=100)
        node_ids = torch.tensor([0, 1, 5, 10, 20], dtype=torch.long)

        output = pe(node_ids)
        assert output.shape == (len(node_ids), 16, 2)  # cos, sin pairs

    def test_pe_factory(self) -> None:
        """Test positional encoding factory."""
        # Laplacian
        pe_lap = PositionalEncodingFactory.create("laplacian", hidden_dim=16)
        assert isinstance(pe_lap, PositionalEncoding)

        # Rotary
        pe_rot = PositionalEncodingFactory.create("rotary", hidden_dim=16)
        assert isinstance(pe_rot, RotaryPositionalEncoding)

        # Invalid
        with pytest.raises(ValueError):
            PositionalEncodingFactory.create("invalid", hidden_dim=16)


class TestAttention:
    """Test attention mechanism modules."""

    def test_scaled_dot_product_attention_shapes(self) -> None:
        """Test scaled dot-product attention shapes."""
        attn = ScaledDotProductAttention(d_k=64)
        batch_size, seq_len = 4, 10

        Q = torch.randn(batch_size, seq_len, 64)
        K = torch.randn(batch_size, seq_len, 64)
        V = torch.randn(batch_size, seq_len, 64)

        output, weights = attn(Q, K, V)
        assert output.shape == V.shape
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_scaled_dot_product_attention_masking(self) -> None:
        """Test attention masking."""
        attn = ScaledDotProductAttention(d_k=64)
        seq_len = 5

        Q = torch.randn(1, seq_len, 64)
        K = torch.randn(1, seq_len, 64)
        V = torch.randn(1, seq_len, 64)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
        mask = mask.unsqueeze(0)

        output, weights = attn(Q, K, V, mask=mask)
        assert output.shape == V.shape
        assert not torch.isnan(output).any()

    def test_multi_head_attention_initialization(self) -> None:
        """Test multi-head attention initialization."""
        mha = MultiHeadAttention(hidden_dim=256, num_heads=8)
        assert mha.hidden_dim == 256
        assert mha.num_heads == 8
        assert mha.d_k == 32

    def test_multi_head_attention_forward(self) -> None:
        """Test multi-head attention forward pass."""
        mha = MultiHeadAttention(hidden_dim=256, num_heads=8)
        batch_size, seq_len = 4, 10

        Q = torch.randn(batch_size, seq_len, 256)
        K = torch.randn(batch_size, seq_len, 256)
        V = torch.randn(batch_size, seq_len, 256)

        output, weights = mha(Q, K, V)
        assert output.shape == (batch_size, seq_len, 256)
        assert weights.shape == (batch_size, seq_len, seq_len)
        assert not torch.isnan(output).any()

    def test_multi_head_attention_graph_mode(self) -> None:
        """Test multi-head attention for graph inputs."""
        mha = MultiHeadAttention(hidden_dim=128, num_heads=4)
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        output, weights = mha(x, x, x, edge_index=edge_index)
        assert output.shape == (num_nodes, 128)


class TestTransformer:
    """Test Graph Transformer components."""

    def test_residual_gating_block(self) -> None:
        """Test residual gating mechanism."""
        from src.models.foundation.transformer import ResidualGatingBlock

        gate = ResidualGatingBlock(hidden_dim=256)
        residual = torch.randn(4, 10, 256)
        block_out = torch.randn(4, 10, 256)

        output = gate(residual, block_out)
        assert output.shape == residual.shape

    def test_feed_forward_network(self) -> None:
        """Test feed-forward network."""
        from src.models.foundation.transformer import FeedForwardNetwork

        ffn = FeedForwardNetwork(hidden_dim=256, ff_dim=1024)
        x = torch.randn(4, 10, 256)

        output = ffn(x)
        assert output.shape == x.shape

    def test_graph_transformer_layer(self) -> None:
        """Test single graph transformer layer."""
        layer = GraphTransformerLayer(hidden_dim=256, num_heads=8)
        num_nodes = 16

        x = torch.randn(num_nodes, 256)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        output, attn = layer(x, edge_index=edge_index)
        assert output.shape == (num_nodes, 256)
        assert attn is not None

    def test_graph_transformer_encoder(self) -> None:
        """Test stacked graph transformer encoder."""
        encoder = GraphTransformerEncoder(
            hidden_dim=256, num_layers=4, num_heads=8
        )
        num_nodes = 16

        x = torch.randn(num_nodes, 256)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        output, attention_list = encoder(
            x, edge_index=edge_index, return_attention=True
        )
        assert output.shape == (num_nodes, 256)
        assert len(attention_list) == 4


class TestLinkPrediction:
    """Test link prediction components."""

    def test_link_prediction_head_initialization(self) -> None:
        """Test link prediction head initialization."""
        lp_head = LinkPredictionHead(hidden_dim=256)
        assert lp_head.hidden_dim == 256
        assert lp_head.aggregation == "hadamard"

    def test_link_prediction_head_forward(self) -> None:
        """Test link prediction head forward pass."""
        lp_head = LinkPredictionHead(hidden_dim=256, aggregation="hadamard")
        num_edges = 10

        source = torch.randn(num_edges, 256)
        target = torch.randn(num_edges, 256)

        output = lp_head(source, target)
        assert output.shape == (num_edges,)
        assert torch.all((output >= 0) & (output <= 1))

    def test_link_prediction_aggregations(self) -> None:
        """Test different aggregation methods."""
        aggregations = ["hadamard", "concat", "sum", "mean"]

        for agg in aggregations:
            lp_head = LinkPredictionHead(hidden_dim=256, aggregation=agg)
            source = torch.randn(10, 256)
            target = torch.randn(10, 256)

            output = lp_head(source, target)
            assert output.shape == (10,)
            assert torch.all((output >= 0) & (output <= 1))

    def test_adapter_module(self) -> None:
        """Test adapter module."""
        adapter = AdapterModule(hidden_dim=256, adapter_dim=64)
        x = torch.randn(16, 256)

        output = adapter(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_edge_scoring_head(self) -> None:
        """Test edge scoring head with adapter."""
        head = EdgeScoringHead(hidden_dim=256, use_adapter=True)
        source = torch.randn(10, 256)
        target = torch.randn(10, 256)

        output = head(source, target)
        assert output.shape == (10,)
        assert torch.all((output >= 0) & (output <= 1))


class TestFoundationModel:
    """Test complete foundation models."""

    def test_graph_foundation_encoder(self) -> None:
        """Test Graph Foundation encoder."""
        encoder = GraphFoundationEncoder(
            input_dim=128, hidden_dim=256, num_layers=4, num_heads=8
        )
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        embeddings, attention = encoder(x, edge_index=edge_index)
        assert embeddings.shape == (num_nodes, 256)

    def test_graph_foundation_link_predictor(self) -> None:
        """Test Graph Foundation link predictor."""
        model = GraphFoundationLinkPredictor(
            input_dim=128, hidden_dim=256, num_layers=4
        )
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        scores = model(x, edge_index)
        assert scores.shape == (4,)
        assert torch.all((scores >= 0) & (scores <= 1))

    def test_graph_foundation_with_adapter(self) -> None:
        """Test Graph Foundation with adapter."""
        model = GraphFoundationWithAdapter(
            input_dim=128, hidden_dim=256, num_layers=4, adapter_dim=64
        )
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        scores, embeddings = model(x, edge_index, return_embeddings=True)
        assert scores.shape == (4,)
        assert embeddings.shape == (num_nodes, 256)


class TestBaseline:
    """Test GraphSAGE baseline models."""

    def test_graphsage_baseline_initialization(self) -> None:
        """Test GraphSAGE initialization."""
        model = GraphSAGEBaseline(input_dim=128, hidden_dim=256, num_layers=3)
        assert model.input_dim == 128
        assert model.hidden_dim == 256
        assert len(model.layers) == 3

    def test_graphsage_forward(self) -> None:
        """Test GraphSAGE forward pass."""
        model = GraphSAGEBaseline(input_dim=128, hidden_dim=256, num_layers=3)
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        embeddings = model(x, edge_index)
        assert embeddings.shape == (num_nodes, 256)

    def test_graphsage_link_prediction(self) -> None:
        """Test GraphSAGE link prediction."""
        model = GraphSAGEBaseline(input_dim=128, hidden_dim=256)
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )
        edge_label_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        scores = model.predict_links(x, edge_index, edge_label_index)
        assert scores.shape == (2,)
        assert torch.all((scores >= 0) & (scores <= 1))

    def test_graphsage_with_adapter(self) -> None:
        """Test GraphSAGE with adapter."""
        model = GraphSAGEWithAdapter(
            input_dim=128, hidden_dim=256, num_layers=3, adapter_dim=64
        )
        num_nodes = 16

        x = torch.randn(num_nodes, 128)
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )
        edge_label_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        scores = model(x, edge_index, edge_label_index)
        assert scores.shape == (2,)


class TestAdapterModules:
    """Test different adapter implementations."""

    def test_bottleneck_adapter(self) -> None:
        """Test bottleneck adapter."""
        adapter = BottleneckAdapter(hidden_dim=256, adapter_dim=64)
        x = torch.randn(16, 256)

        output = adapter(x)
        assert output.shape == x.shape

    def test_compact_adapter(self) -> None:
        """Test compact adapter."""
        adapter = CompactAdapter(hidden_dim=256, rank=8)
        x = torch.randn(16, 256)

        output = adapter(x)
        assert output.shape == x.shape

    def test_lora_adapter(self) -> None:
        """Test LoRA adapter."""
        adapter = LoRAAdapter(hidden_dim=256, rank=8, alpha=16.0)
        x = torch.randn(16, 256)

        output = adapter(x)
        assert output.shape == x.shape

    def test_adapter_module_factory(self) -> None:
        """Test adapter module factory."""
        types = ["bottleneck", "compact", "lora"]

        for atype in types:
            adapter = AdapterModuleFactory(hidden_dim=256, adapter_type=atype)
            x = torch.randn(16, 256)
            output = adapter(x)
            assert output.shape == x.shape

    def test_adapter_parameter_counts(self) -> None:
        """Test adapter parameter counting."""
        hidden_dim = 256

        bottleneck_params = AdapterModuleFactory.count_parameters(
            hidden_dim, "bottleneck", adapter_dim=64
        )
        compact_params = AdapterModuleFactory.count_parameters(
            hidden_dim, "compact", rank=8
        )
        lora_params = AdapterModuleFactory.count_parameters(
            hidden_dim, "lora", rank=8
        )

        assert bottleneck_params > compact_params
        assert compact_params > lora_params


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_foundation_pipeline(self) -> None:
        """Test complete foundation model pipeline."""
        model = GraphFoundationLinkPredictor(
            input_dim=128, hidden_dim=256, num_layers=4, num_heads=8
        )
        num_nodes = 32

        x = torch.randn(num_nodes, 128)
        edge_index = torch.randint(0, num_nodes, (2, 64))

        # Forward pass
        scores = model(x, edge_index)
        assert scores.shape == (64,)
        assert not torch.isnan(scores).any()

    def test_baseline_pipeline(self) -> None:
        """Test GraphSAGE pipeline."""
        model = GraphSAGEBaseline(input_dim=128, hidden_dim=256, num_layers=3)
        num_nodes = 32

        x = torch.randn(num_nodes, 128)
        edge_index = torch.randint(0, num_nodes, (2, 64))

        scores = model.predict_links(x, edge_index, edge_index[:, :10])
        assert scores.shape == (10,)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through models."""
        model = GraphFoundationLinkPredictor(input_dim=128, hidden_dim=256)
        loss_fn = nn.BCELoss()

        x = torch.randn(16, 128, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_labels = torch.ones(3)

        scores = model(x, edge_index)
        loss = loss_fn(scores, edge_labels)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert model.encoder.base_model.input_projection.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
