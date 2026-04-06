"""Phase 4 Evaluation pipeline."""

import torch
from pathlib import Path
from typing import Dict

from src.config_loader import load_config, create_config_parser
from src.data.pipeline import DataPipeline
from src.evaluation.metrics import compute_roc_auc, compute_pr_auc
from src.models.baseline.model import GraphSAGEBaseline
from src.models import GraphFoundationModel
from src.training.trainer import MetricsCalculator


def _eval_model(model, graph, device):
    model.eval()
    with torch.no_grad():
        out = model(graph)
        node_emb = out['node_emb']
        edge_index = graph.edge_index.to(device) if hasattr(graph, 'edge_index') else graph.edge_index.to(device)
        num_edges = edge_index.shape[1]
        half = num_edges // 2
        pos_edges = edge_index[:, :half]
        neg_src = torch.randint(0, node_emb.shape[0], (half,), device=device)
        neg_dst = torch.randint(0, node_emb.shape[0], (half,), device=device)
        neg_edges = torch.stack([neg_src, neg_dst])

        if hasattr(model, 'link_predictor'):
            pos_scores = model.link_predictor(node_emb, pos_edges)
            neg_scores = model.link_predictor(node_emb, neg_edges)
        else:
            row_p, col_p = pos_edges
            row_n, col_n = neg_edges
            pos_scores = torch.sigmoid((node_emb[row_p] * node_emb[col_p]).sum(-1))
            neg_scores = torch.sigmoid((node_emb[row_n] * node_emb[col_n]).sum(-1))

        scores = torch.cat([pos_scores, neg_scores]).squeeze()
        labels = torch.cat([torch.ones(half, device=device), torch.zeros(half, device=device)])
        return MetricsCalculator.compute_metrics(scores, labels)


def main():
    parser = create_config_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline = DataPipeline(config)
    target_graph = pipeline.load_target_domain()

    results = {}

    # Baseline
    baseline = GraphSAGEBaseline(config_path=args.config, device=str(device)).to(device)
    results['baseline'] = _eval_model(baseline, target_graph, device)

    # Foundation
    foundation = GraphFoundationModel(config_path=args.config, device=device)
    results['foundation'] = _eval_model(foundation, target_graph, device)

    print('=== EVALUATION RESULTS ===')
    for method, metrics in results.items():
        print(f'{method.upper()}: ROC-AUC={metrics["roc_auc"]:.3f}, '
              f'PR-AUC={metrics["pr_auc"]:.3f}, F1={metrics["f1"]:.3f}')

    # Save results
    import json
    out_dir = Path(config.get('evaluation.output_dir', 'evaluation/results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_dir}/metrics.json')

    return results


if __name__ == '__main__':
    main()
