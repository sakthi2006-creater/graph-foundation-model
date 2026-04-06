"""PR-AUC metric for link prediction evaluation."""

import torch
from torch import Tensor
from sklearn.metrics import average_precision_score


def compute_pr_auc(scores: Tensor, labels: Tensor) -> float:
    """Compute PR-AUC score."""
    return float(average_precision_score(labels.cpu().numpy(), scores.cpu().numpy()))
