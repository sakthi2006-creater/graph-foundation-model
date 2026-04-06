"""Quick smoke test for all project components."""
import sys

def test(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("\n=== Graph Foundation Model - Smoke Test ===\n")

# 1. Config
print("[1] Config Loader")
def t_config():
    from src.config_loader import load_config
    c = load_config()
    return f"seed={c.get('seed')}, device={c.get('device')}"
test("load_config", t_config)

# 2. Data pipeline
print("\n[2] Data Pipeline")
def t_data():
    from src.config_loader import load_config
    from src.data.pipeline import DataPipeline
    c = load_config()
    p = DataPipeline(c)
    g = p.load_dataset('cora')
    return f"{g.num_nodes} nodes, {g.num_edges} edges, {g.num_features} features"
test("load cora dataset", t_data)

def t_synthetic():
    from src.data.loaders.synthetic_loader import SyntheticGraphLoader
    loader = SyntheticGraphLoader(num_nodes=50, num_features=32)
    g = loader.load()
    return f"{g.num_nodes} nodes, {g.num_edges} edges"
test("synthetic graph loader", t_synthetic)

# 3. Models
print("\n[3] Models")
def t_foundation():
    from src.models import GraphFoundationModel
    from src.data.loaders.synthetic_loader import SyntheticGraphLoader
    model = GraphFoundationModel()
    g = SyntheticGraphLoader(num_nodes=50, num_features=32).load()
    out = model(g)
    emb = out['node_emb']
    return f"node_emb shape={list(emb.shape)}"
test("GraphFoundationModel forward", t_foundation)

def t_baseline():
    from src.models.baseline.model import GraphSAGEBaseline
    from src.data.loaders.synthetic_loader import SyntheticGraphLoader
    model = GraphSAGEBaseline(hidden_dim=64)
    g = SyntheticGraphLoader(num_nodes=50, num_features=64).load()
    out = model(g)
    return f"node_emb shape={list(out['node_emb'].shape)}"
test("GraphSAGEBaseline forward", t_baseline)

def t_link_pred():
    import torch
    from src.models import GraphFoundationModel
    from src.data.loaders.synthetic_loader import SyntheticGraphLoader
    model = GraphFoundationModel()
    g = SyntheticGraphLoader(num_nodes=50, num_features=32).load()
    out = model(g)
    emb = out['node_emb']
    query = torch.randint(0, 50, (2, 10))
    scores = model.link_predictor(emb, query)
    return f"link scores shape={list(scores.shape)}, range=[{scores.min():.2f},{scores.max():.2f}]"
test("LinkPredictor", t_link_pred)

# 4. Training
print("\n[4] Training (1 step)")
def t_pretrain():
    from src.training.pretrain.pretrain_model import PretrainTrainer
    trainer = PretrainTrainer()
    trainer.fit(num_epochs=1)
    return "1 epoch complete"
test("PretrainTrainer (1 epoch)", t_pretrain)

def t_maml():
    from src.training.meta_learning.maml_trainer import MAMLTrainer
    trainer = MAMLTrainer()
    trainer.meta_fit(num_epochs=1, source_domains=['cora'])
    return "1 epoch complete"
test("MAMLTrainer (1 epoch)", t_maml)

def t_finetune():
    from src.training.finetune.finetune_model import FinetuneTrainer
    trainer = FinetuneTrainer()
    trainer.fit(num_epochs=1)
    return "1 epoch complete"
test("FinetuneTrainer (1 epoch)", t_finetune)

# 5. Evaluation
print("\n[5] Evaluation")
def t_eval():
    import torch
    from src.training.trainer import MetricsCalculator
    scores = torch.rand(100)
    labels = torch.randint(0, 2, (100,)).float()
    m = MetricsCalculator.compute_metrics(scores, labels)
    return f"roc_auc={m['roc_auc']:.3f}, f1={m['f1']:.3f}"
test("MetricsCalculator", t_eval)

# 6. Backend
print("\n[6] FastAPI Backend")
def t_backend():
    from src.backend.app import app
    return f"routes={[r.path for r in app.routes]}"
test("FastAPI app import", t_backend)

print("\n=== Done! ===")
print("\nTo run the services:")
print("  FastAPI:   python -m uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload")
print("  Streamlit: streamlit run src/streamlit_app/app.py")
