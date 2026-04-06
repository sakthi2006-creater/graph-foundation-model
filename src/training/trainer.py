"""Base training utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config_loader import load_config
from src.utils import set_seed, get_device, setup_logging, ensure_dir, log_environment_info


class MetricsCalculator:
    """Compute link prediction metrics."""

    @staticmethod
    def compute_metrics(scores: Tensor, labels: Tensor) -> Dict[str, float]:
        scores_cpu = scores.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()

        roc_auc = roc_auc_score(labels_cpu, scores_cpu)
        pr_auc = average_precision_score(labels_cpu, scores_cpu)

        fpr, tpr, thresholds = roc_curve(labels_cpu, scores_cpu)
        youden_j = tpr - fpr
        best_thresh = thresholds[youden_j.argmax()]
        preds = (scores_cpu >= best_thresh).astype(int)

        tp = ((preds == 1) & (labels_cpu == 1)).sum()
        fp = ((preds == 1) & (labels_cpu == 0)).sum()
        fn = ((preds == 0) & (labels_cpu == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1': float(f1),
            'best_threshold': float(best_thresh),
        }


class BaseTrainer:
    """Base class for all training pipelines."""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_loader = load_config(config_path)
        self.config = self.config_loader.config
        self.seed = self.config.get('seed', 42)
        self.device = get_device(self.config.get('device', 'auto'))

        set_seed(self.seed)
        setup_logging(level='INFO')

        self.run_dir = Path('checkpoints')
        self.run_dir.mkdir(exist_ok=True)
        self.global_step = 0

        # Optional wandb
        if self.config.get('logging', {}).get('use_wandb', False):
            try:
                import wandb
                wandb.init(
                    project=self.config.get('logging', {}).get('wandb_project', 'graph-foundation'),
                    config=self.config_loader.to_dict(),
                )
            except Exception:
                pass

    def setup_optimizer(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        opt_cfg = self.config.get('optimizer', {})
        lr = float(opt_cfg.get('lr', 1e-4))
        wd = float(opt_cfg.get('weight_decay', 1e-4))
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(self.config.get('training', {}).get('max_epochs', 50)),
            eta_min=float(self.config.get('scheduler', {}).get('eta_min', 1e-6)),
        )
        return optimizer, scheduler

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, **kwargs):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **kwargs,
        }
        path = self.run_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(ckpt, path)
        logger.info(f'Saved checkpoint: {path}')

    def log_metrics(self, metrics: Dict[str, float]):
        self.global_step += 1
        logger.info(f'Step {self.global_step}: {metrics}')

    def compute_loss(self, model: nn.Module, batch: Any, is_train: bool = True) -> Tensor:
        raise NotImplementedError

    def collect_metrics(self, model: nn.Module, batch: Any) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
