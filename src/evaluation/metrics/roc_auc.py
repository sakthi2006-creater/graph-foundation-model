"""ROC-AUC metric for link prediction evaluation."""

import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score


def compute_roc_auc(scores: Tensor, labels: Tensor) -> float:
    """Compute ROC-AUC score."""
    return float(roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))
