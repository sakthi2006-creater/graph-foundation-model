"""Unit tests for Phase 2 models."""

import pytest
import torch
from src.models.positional_encoding import PositionalEncoder
from src.models.foundation.model import GraphFoundationModel
from src.models.foundation.attention import MultiHeadAttention
from src.models.foundation.transformer import GraphTransformerLayer
from src.models.foundation.link_predictor import LinkPredictor
from src.models.baseline.model import GraphSAGEBaseline
from src.data.loaders.synthetic_loader import SyntheticGraphLoader
from src.config_loader import load_config

@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def synthetic_data():
    loader = SyntheticGraphLoader(load_config())
    return loader.generate_graphs(1)[0]

def test_positional_encoder(config):
    encoder = PositionalEncoder(config_path='config.yaml')
    edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    pe = encoder(edge_index)
    assert pe.shape == (3, config.model.foundation.laplacian_pe_dim)
    assert pe.min() >= -2 and pe.max() <= 2

def test_multihead_attention(synthetic_data):
    att = MultiHeadAttention(64, 64, num_heads=4)
    out = att(synthetic_data.x[:10], synthetic_data.edge_index[:, :20])
    assert out.shape == (10, 64)

def test_transformer_layer(synthetic_data):
    layer = GraphTransformerLayer(64, num_heads=4)
    out = layer(synthetic_data.x[:10], synthetic_data.edge_index[:, :20])
    assert out.shape == synthetic_data.x[:10].shape

def test_link_predictor(synthetic_data):
    predictor = LinkPredictor(64)
    scores = predictor(synthetic_data.x[:10], synthetic_data.edge_index[:, :5])
    assert scores.shape == (5,)

def test_graph_foundation_model(synthetic_data):
    model = GraphFoundationModel()
    outputs = model(synthetic_data)
    assert 'node_emb' in outputs
    assert outputs['node_emb'].shape[1] == 128
    
    # Test link prediction
    query_edges = synthetic_data.edge_index[:, :10]
    outputs_query = model(synthetic_data, query_edges)
    assert 'link_scores' in outputs_query
    assert outputs_query['link_scores'].shape == (10,)

def test_graphsage_baseline(synthetic_data):
    model = GraphSAGEBaseline()
    outputs = model(synthetic_data)
    assert 'node_emb' in outputs
    assert outputs['node_emb'].shape[1] == 128

@pytest.mark.parametrize("model_class", [GraphFoundationModel, GraphSAGEBaseline])
def test_model_device(model_class):
    model = model_class()
    assert next(model.parameters()).device.type in ['cpu', 'cuda']

def test_model_forward_backward(synthetic_data):
    model = GraphFoundationModel()
    outputs = model(synthetic_data)
    loss = outputs['node_emb'].sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

