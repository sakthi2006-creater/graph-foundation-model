"""Pretraining with masked edge reconstruction."""

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Tuple, Any

import torch.nn.functional as F

from src.data.pipeline import DataPipeline
from src.training.trainer import BaseTrainer
from src.models import GraphFoundationModel


class PretrainTrainer(BaseTrainer):
    """Pretraining trainer for masked edge prediction."""

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.model = GraphFoundationModel(config_path=config_path, device=self.device)
        self.pretrain_cfg = self.config.get('pretraining', {})

        pipeline = DataPipeline(self.config_loader)
        source_graphs = list(pipeline.load_all_source_domains().values())
        self.train_graphs = source_graphs
        self.val_graphs = source_graphs  # simplified

    def mask_edges(self, edge_index: Tensor, mask_rate: float = 0.2):
        num_edges = edge_index.shape[1]
        num_mask = max(1, int(num_edges * mask_rate))
        mask_indices = torch.randperm(num_edges, device=edge_index.device)[:num_mask]
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        keep_mask[mask_indices] = False
        return edge_index[:, keep_mask], edge_index[:, mask_indices], mask_indices, keep_mask

    def compute_loss(self, model: nn.Module, batch: Any, is_train: bool = True) -> Tensor:
        from src.data.graph_data import GraphData
        data = batch
        if hasattr(data, 'node_features'):
            x, edge_index = data.node_features.to(self.device), data.edge_index.to(self.device)
        else:
            x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        masked_edge_index, orig_masked, _, _ = self.mask_edges(edge_index)

        if hasattr(data, 'node_features'):
            masked_data = GraphData(node_features=x, edge_index=masked_edge_index)
        else:
            from torch_geometric.data import Data
            masked_data = Data(x=x, edge_index=masked_edge_index)

        node_emb = model(masked_data)['node_emb']
        link_scores = model.link_predictor(node_emb, orig_masked)
        mask_labels = torch.ones(orig_masked.shape[1], dtype=torch.float, device=self.device)
        return F.binary_cross_entropy(link_scores.squeeze(-1), mask_labels)

    def collect_metrics(self, model: nn.Module, batch: Any) -> Tuple[Tensor, Tensor]:
        if hasattr(batch, 'node_features'):
            x, edge_index = batch.node_features.to(self.device), batch.edge_index.to(self.device)
        else:
            x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)
        node_emb = model(batch)['node_emb']
        scores = model.link_predictor(node_emb, edge_index).squeeze(-1)
        labels = torch.ones_like(scores)
        return scores, labels

    def fit(self, num_epochs: int = 1):
        """Simple training loop over source graphs."""
        optimizer, scheduler = self.setup_optimizer(self.model)
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for graph in self.train_graphs:
                optimizer.zero_grad()
                loss = self.compute_loss(self.model, graph)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(self.train_graphs)
            scheduler.step()
            from loguru import logger
            logger.info(f'[PRETRAIN] Epoch {epoch+1}/{num_epochs} | Loss: {avg:.4f}')
        self.save_checkpoint(self.model, optimizer, num_epochs - 1)
