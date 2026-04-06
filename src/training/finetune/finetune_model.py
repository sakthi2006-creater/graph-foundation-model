"""Adapter-based few-shot finetuning with frozen foundation encoder."""

import torch
from torch import nn, Tensor
from typing import Any, Optional, Tuple

from src.data.pipeline import DataPipeline
from src.models import GraphFoundationModel
from src.models.adapter.model import BottleneckAdapter
from src.training.trainer import BaseTrainer, MetricsCalculator
from loguru import logger


class FinetuneTrainer(BaseTrainer):
    """Finetune pretrained foundation model with lightweight adapter."""

    def __init__(self, config_path: str = 'config.yaml', pretrained_ckpt: Optional[str] = None):
        super().__init__(config_path)

        self.foundation = GraphFoundationModel(config_path=config_path, device=self.device)
        if pretrained_ckpt:
            ckpt = torch.load(pretrained_ckpt, map_location=self.device)
            self.foundation.load_state_dict(ckpt['model_state_dict'])
            logger.info(f'Loaded pretrained from {pretrained_ckpt}')

        for param in self.foundation.parameters():
            param.requires_grad_(False)

        hidden_dim = self.config.get('model.foundation.hidden_dim', 128)
        adapter_dim = self.config.get('model.adapter.bottleneck_dim', 64)
        self.adapter = BottleneckAdapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim).to(self.device)
        self.model = self.foundation  # forward uses foundation + adapter inline

        self.optimizer, self.scheduler = self.setup_optimizer(self.adapter)

        pipeline = DataPipeline(self.config_loader)
        self.target_graph = pipeline.load_dataset(
            self.config.get('data.target_domain', 'amazon_photo'))

    def compute_loss(self, model: nn.Module, batch: Any, is_train: bool = True) -> Tensor:
        if hasattr(batch, 'node_features'):
            x, edge_index = batch.node_features.to(self.device), batch.edge_index.to(self.device)
        else:
            x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)

        with torch.no_grad():
            node_emb = self.foundation(batch)['node_emb']

        adapted_emb = self.adapter(node_emb)

        num_edges = edge_index.shape[1]
        half = num_edges // 2
        pos_edges = edge_index[:, :half]
        neg_src = torch.randint(0, x.shape[0], (half,), device=self.device)
        neg_dst = torch.randint(0, x.shape[0], (half,), device=self.device)
        neg_edges = torch.stack([neg_src, neg_dst])

        pos_scores = self.foundation.link_predictor(adapted_emb, pos_edges)
        neg_scores = self.foundation.link_predictor(adapted_emb, neg_edges)
        pos_labels = torch.ones(half, device=self.device)
        neg_labels = torch.zeros(half, device=self.device)

        return torch.nn.functional.binary_cross_entropy(
            torch.cat([pos_scores, neg_scores]).squeeze(),
            torch.cat([pos_labels, neg_labels])
        )

    def collect_metrics(self, model: nn.Module, batch: Any) -> Tuple[Tensor, Tensor]:
        if hasattr(batch, 'node_features'):
            edge_index = batch.edge_index.to(self.device)
        else:
            edge_index = batch.edge_index.to(self.device)
        with torch.no_grad():
            node_emb = self.foundation(batch)['node_emb']
        adapted_emb = self.adapter(node_emb)
        scores = self.foundation.link_predictor(adapted_emb, edge_index).squeeze()
        labels = torch.ones(scores.shape[0], device=self.device)
        return scores, labels

    def fit(self, num_epochs: int = 1):
        """Finetuning loop (adapter only)."""
        logger.info('Starting adapter finetuning...')
        for epoch in range(num_epochs):
            self.adapter.train()
            self.optimizer.zero_grad()
            loss = self.compute_loss(None, self.target_graph)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            logger.info(f'[FINETUNE] Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}')
        self.save_checkpoint(self.adapter, self.optimizer, num_epochs - 1)
