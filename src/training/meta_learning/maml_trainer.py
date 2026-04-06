"""1st-order MAML Trainer for few-shot adaptation across domains."""

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple, Any

from src.data.pipeline import DataPipeline
from src.models import GraphFoundationModel
from src.training.trainer import BaseTrainer, MetricsCalculator
from loguru import logger


class MAMLTrainer(BaseTrainer):
    """First-order MAML for graph meta-learning."""

    def __init__(self, config_path: str = 'config.yaml', inner_steps: int = 5):
        super().__init__(config_path)
        self.inner_steps = inner_steps
        self.inner_lr = self.config.get('meta_learning.inner_lr', 1e-3)
        self.model = GraphFoundationModel(config_path=config_path, device=self.device)
        self.meta_optimizer, self.meta_scheduler = self.setup_optimizer(self.model)
        self.pipeline = DataPipeline(self.config_loader)

    def compute_loss(self, model: nn.Module, batch: Any, is_train: bool = True) -> Tensor:
        if hasattr(batch, 'node_features'):
            x, edge_index = batch.node_features.to(self.device), batch.edge_index.to(self.device)
        else:
            x, edge_index = batch.x.to(self.device), batch.edge_index.to(self.device)

        node_emb = model(batch)['node_emb']
        num_edges = edge_index.shape[1]
        half = num_edges // 2
        pos_edges = edge_index[:, :half]
        neg_src = torch.randint(0, x.shape[0], (half,), device=self.device)
        neg_dst = torch.randint(0, x.shape[0], (half,), device=self.device)
        neg_edges = torch.stack([neg_src, neg_dst])

        pos_scores = model.link_predictor(node_emb, pos_edges)
        neg_scores = model.link_predictor(node_emb, neg_edges)
        pos_labels = torch.ones(half, device=self.device)
        neg_labels = torch.zeros(half, device=self.device)

        loss = torch.nn.functional.binary_cross_entropy(
            torch.cat([pos_scores, neg_scores]).squeeze(),
            torch.cat([pos_labels, neg_labels])
        )
        return loss

    def collect_metrics(self, model: nn.Module, batch: Any) -> Tuple[Tensor, Tensor]:
        if hasattr(batch, 'node_features'):
            edge_index = batch.edge_index.to(self.device)
        else:
            edge_index = batch.edge_index.to(self.device)
        node_emb = model(batch)['node_emb']
        scores = model.link_predictor(node_emb, edge_index).squeeze()
        labels = torch.ones(scores.shape[0], device=self.device)
        return scores, labels

    def meta_fit(self, num_epochs: int = 1, source_domains: Optional[List[str]] = None):
        """Full meta-training loop."""
        if source_domains is None:
            source_domains = self.config.get('data.source_domains', ['cora', 'pubmed', 'citeseer'])

        source_graphs = {d: self.pipeline.load_dataset(d) for d in source_domains}

        for epoch in range(num_epochs):
            meta_losses = []
            for src_name, src_graph in source_graphs.items():
                self.model.train()
                loss = self.compute_loss(self.model, src_graph)
                meta_losses.append(loss)

            total_meta_loss = sum(meta_losses) / len(meta_losses)
            self.meta_optimizer.zero_grad()
            total_meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.meta_optimizer.step()
            self.meta_scheduler.step()

            logger.info(f'[MAML] Epoch {epoch+1}/{num_epochs} | Meta Loss: {total_meta_loss.item():.4f}')

        self.save_checkpoint(self.model, self.meta_optimizer, num_epochs - 1)
