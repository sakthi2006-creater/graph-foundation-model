################################################################################
# GRAPH FOUNDATION MODEL - PROJECT STRUCTURE
# Complete file organization for production-ready system
################################################################################

📦 santha/ (Project Root)
│
├── 📄 config.yaml                      # Master configuration (hyperparameters)
├── 📄 requirements.txt                 # Production dependencies (pinned)
├── 📄 README.md                        # Project documentation & quickstart
├── 📄 .gitignore                       # Git ignore rules
├── 📄 .env.example                     # Environment variables template
│
│
├── 📂 src/                             # Main source code
│   ├── 📄 __init__.py                  # Package root
│   ├── 📄 utils.py                     # Helper functions (device, seed, paths)
│   ├── 📄 config_loader.py             # YAML config parsing
│   │
│   ├── 📂 data/                        # Data pipeline
│   │   ├── 📄 __init__.py
│   │   ├── 📄 graph_data.py            # GraphData class definition
│   │   ├── 📂 loaders/                 # Dataset loading
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 base_loader.py       # BaseGraphLoader (abstract)
│   │   │   ├── 📄 pyg_loader.py        # PyGDatasetLoader (Cora, PubMed, etc.)
│   │   │   ├── 📄 synthetic_loader.py  # SyntheticGraphLoader (fallback)
│   │   │
│   │   ├── 📂 processors/              # Data processing
│   │       ├── 📄 __init__.py
│   │       ├── 📄 normalizer.py        # Feature normalization
│   │       ├── 📄 splitter.py          # Edge splitting (train/val/test)
│   │       ├── 📄 sampler.py           # Negative sampling (degree-weighted)
│   │       ├── 📄 validator.py         # Graph schema validation
│   │
│   ├── 📂 models/                      # Neural network models
│   │   ├── 📄 __init__.py
│   │   ├── 📄 positional_encoding.py   # Laplacian PE, Rotary PE
│   │   │
│   │   ├── 📂 foundation/              # Graph Foundation Model
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 transformer.py       # GraphTransformerLayer
│   │   │   ├── 📄 attention.py         # Multi-head attention
│   │   │   ├── 📄 link_predictor.py    # Link prediction MLP head
│   │   │   ├── 📄 model.py             # Full GraphFoundationModel
│   │   │
│   │   ├── 📂 baseline/                # GraphSAGE baseline
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 model.py             # GraphSAGEBaseline
│   │   │
│   │   ├── 📂 adapter/                 # Few-shot adapter
│   │       ├── 📄 __init__.py
│   │       ├── 📄 model.py             # AdapterModule (bottleneck MLP)
│   │
│   ├── 📂 training/                    # Training loops
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_trainer.py          # BaseTrainer (common logic)
│   │   ├── 📄 losses.py                # Loss functions, class balancing
│   │   ├── 📄 callbacks.py             # WandB, checkpointing, early stopping
│   │   │
│   │   ├── 📂 pretrain/                # Pretraining (masked edge prediction)
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 trainer.py           # PretrainTrainer
│   │   │   ├── 📄 main.py              # Entry point: python -m src.training.pretrain.main
│   │   │
│   │   ├── 📂 meta_learning/           # MAML meta-learning
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 maml.py              # MAML loop (inner/outer)
│   │   │   ├── 📄 task_sampler.py      # Sample tasks from graphs
│   │   │   ├── 📄 trainer.py           # MetaLearningTrainer
│   │   │   ├── 📄 main.py              # Entry point: python -m src.training.meta_learning.main
│   │   │
│   │   ├── 📂 finetune/                # Few-shot fine-tuning
│   │       ├── 📄 __init__.py
│   │       ├── 📄 trainer.py           # FinetuneTrainer (adapter-only)
│   │       ├── 📄 main.py              # Entry point: python -m src.training.finetune.main
│   │
│   ├── 📂 evaluation/                  # Evaluation & metrics
│   │   ├── 📄 __init__.py
│   │   ├── 📄 evaluator.py             # Main Evaluator class
│   │   │
│   │   ├── 📂 metrics/                 # Metric computations
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 roc_auc.py           # ROC-AUC computation
│   │   │   ├── 📄 pr_auc.py            # PR-AUC computation
│   │   │   ├── 📄 threshold.py         # Youden J-statistic, optimal threshold
│   │   │   ├── 📄 confusion.py         # Confusion matrix
│   │   │   ├── 📄 calibration.py       # Calibration metrics
│   │   │
│   │   ├── 📂 visualization/           # Plotting & visualization
│   │       ├── 📄 __init__.py
│   │       ├── 📄 plotly_viz.py        # Plotly interactive plots
│   │       ├── 📄 matplotlib_viz.py    # Static matplotlib plots
│   │       ├── 📄 d3js_utils.py        # D3.js data export
│   │       ├── 📄 graph_plotter.py     # Graph structure visualization
│   │
│   ├── 📂 backend/                     # FastAPI backend
│   │   ├── 📄 __init__.py
│   │   ├── 📄 app.py                   # FastAPI app setup & CORS
│   │   ├── 📄 config.py                # Backend config
│   │   ├── 📄 dependencies.py          # Dependency injection (model loading)
│   │   │
│   │   ├── 📂 routes/                  # API endpoint handlers
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 datasets.py          # GET /dataset/{name}
│   │   │   ├── 📄 training.py          # POST /train/*
│   │   │   ├── 📄 predict.py           # POST /predict/link
│   │   │   ├── 📄 results.py           # GET /results/*
│   │   │   ├── 📄 health.py            # GET /health
│   │   │
│   │   ├── 📂 schemas/                 # Pydantic models
│   │       ├── 📄 __init__.py
│   │       ├── 📄 dataset.py           # DatasetSchema, GraphMetadata
│   │       ├── 📄 training.py          # TrainRequest, TrainResponse
│   │       ├── 📄 prediction.py        # PredictionRequest, PredictionResponse
│   │       ├── 📄 results.py           # MetricsResponse, EmbeddingsResponse
│   │
│   ├── 📂 streamlit_app/               # Streamlit dashboard
│   │   ├── 📄 __init__.py
│   │   ├── 📄 app.py                   # Main Streamlit app
│   │   ├── 📄 config.py                # Streamlit config & utils
│   │   │
│   │   ├── 📂 pages/                   # Multi-page Streamlit app
│   │       ├── 📄 __init__.py
│   │       ├── 📄 1_Graph_Explorer.py  # Page 1: Dataset exploration
│   │       ├── 📄 2_Link_Prediction.py # Page 2: Link prediction interface
│   │       ├── 📄 3_Results.py         # Page 3: Results dashboard
│   │       ├── 📄 4_Training.py        # Page 4: Training insights
│   │       ├── 📄 5_Embeddings.py      # Page 5: t-SNE explorer
│   │
│   ├── 📂 visualization/               # Visualization utilities
│   │   ├── 📄 __init__.py
│   │   │
│   │   ├── 📂 plotly/                  # Plotly interactive plots
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 roc_curves.py
│   │   │   ├── 📄 pr_curves.py
│   │   │   ├── 📄 training_loss.py
│   │   │   ├── 📄 tsne_explorer.py
│   │   │
│   │   ├── 📂 d3js/                    # D3.js components
│   │       ├── 📄 __init__.py
│   │       ├── 📄 graph_visualizer.js  # D3 force-directed layout
│   │       ├── 📄 data_exporter.py     # Export graph data for D3
│   │
│
├── 📂 tests/                           # Test suite
│   ├── 📄 __init__.py
│   ├── 📄 conftest.py                  # Pytest fixtures
│   │
│   ├── 📂 unit/                        # Unit tests
│   │   ├── 📄 __init__.py
│   │   ├── 📄 test_data_loaders.py     # Test dataset loading
│   │   ├── 📄 test_processors.py       # Test normalization, splitting
│   │   ├── 📄 test_models.py           # Test model forward pass
│   │   ├── 📄 test_losses.py           # Test loss functions
│   │   ├── 📄 test_metrics.py          # Test evaluation metrics
│   │
│   ├── 📂 integration/                 # Integration tests
│       ├── 📄 __init__.py
│       ├── 📄 test_pretrain_pipeline.py   # Full pretraining pipeline
│       ├── 📄 test_meta_learning.py       # MAML pipeline
│       ├── 📄 test_finetune_pipeline.py   # Fine-tuning pipeline
│       ├── 📄 test_api.py                 # FastAPI endpoints
│
├── 📂 notebooks/                       # Jupyter notebooks
│   ├── 📂 exploration/                 # EDA & exploration
│   │   ├── 📄 01_dataset_statistics.ipynb
│   │   ├── 📄 02_graph_visualization.ipynb
│   │   ├── 📄 03_feature_analysis.ipynb
│   │
│   ├── 📂 analysis/                    # Analysis & results
│       ├── 📄 01_pretraining_results.ipynb
│       ├── 📄 02_meta_learning_analysis.ipynb
│       ├── 📄 03_model_comparison.ipynb
│
├── 📂 docs/                            # Documentation
│   ├── 📄 README.md → see root
│   │
│   ├── 📂 api/                         # API documentation
│   │   ├── 📄 endpoints.md             # All API endpoints
│   │   ├── 📄 examples.md              # cURL / Python examples
│   │   ├── 📄 openapi.json             # OpenAPI spec (auto-generated)
│   │
│   ├── 📂 architecture/                # System architecture
│   │   ├── 📄 system_overview.md       # High-level architecture
│   │   ├── 📄 data_pipeline.md         # Data flow diagram
│   │   ├── 📄 model_architecture.md    # Model design details
│   │   ├── 📄 experiment_flow.md       # Training pipeline flow
│   │   ├── 📄 architecture.png         # Mermaid diagram (exported)
│   │
│   ├── 📂 guides/                      # User guides
│       ├── 📄 quickstart.md            # Get started in 5 minutes
│       ├── 📄 installation.md          # Detailed setup
│       ├── 📄 training.md              # Training guide
│       ├── 📄 evaluation.md            # Evaluation guide
│       ├── 📄 deployment.md            # Docker & production
│
├── 📂 docker/                          # Docker configurations
│   ├── 📄 Dockerfile.backend           # Python backend image
│   ├── 📄 Dockerfile.frontend          # Node.js frontend image
│   ├── 📄 docker-compose.yml           # Multi-container orchestration
│   ├── 📄 .dockerignore                # Docker ignore rules
│
├── 📂 .github/                         # GitHub integration
│   └── 📂 workflows/                   # GitHub Actions CI/CD
│       ├── 📄 lint.yml                 # Lint checks (ruff, eslint)
│       ├── 📄 test.yml                 # Run pytest & jest
│       ├── 📄 docker-build.yml         # Build Docker images
│       ├── 📄 deploy.yml               # Deploy to production
│
├── 📂 data/                            # Data directory
│   └── 📂 cache/                       # Downloaded datasets cache
│
├── 📂 checkpoints/                     # Model checkpoints
│   ├── 📄 foundation_model_v1.pt       # Pretraining checkpoint
│   ├── 📄 foundation_model_finetuned.pt # Fine-tuned checkpoint
│
├── 📂 evaluation/                      # Evaluation results
│   └── 📂 results/
│       ├── 📄 metrics.json             # Evaluation metrics
│       ├── 📄 roc_curve.png            # ROC curve plot
│       ├── 📄 pr_curve.html            # PR curve (interactive)
│       ├── 📄 confusion_matrix.png     # Confusion matrix heatmap
│
├── 📂 logs/                            # Training logs
│   ├── 📄 training.csv                 # CSV log backup
│   └── 📄 error.log                    # Error log
│
├── 📄 .gitignore                       # Git ignore patterns
├── 📄 .env.example                     # Environment template
├── 📄 pyproject.toml                   # Python project metadata
├── 📄 setup.py                         # Package installation config
│

################################################################################
# FILE DESCRIPTIONS & ENTRY POINTS
################################################################################

KEY ENTRY POINTS (Command-line interfaces):
────────────────────────────────────────────────────────────────────────────

1. PRETRAINING (Masked edge prediction on multiple domains)
   Command: python -m src.training.pretrain.main --config config.yaml
   File:    src/training/pretrain/main.py
   Trains:  GraphFoundationModel on source domains (Cora, PubMed, CiteSeer)
   Output:  checkpoints/foundation_model_pretrained.pt

2. META-LEARNING (MAML adaptation loop)
   Command: python -m src.training.meta_learning.main --config config.yaml
   File:    src/training/meta_learning/main.py
   Trains:  Inner loop + outer loop over task batches
   Output:  checkpoints/foundation_model_metalearned.pt

3. FINE-TUNING (Few-shot on target domain)
   Command: python -m src.training.finetune.main --config config.yaml
   File:    src/training/finetune/main.py
   Trains:  Only adapter weights; freezes encoder & predictor
   Output:  checkpoints/foundation_model_finetuned.pt, evaluation/results/

4. EVALUATION & VISUALIZATION
   Command: python -m src.evaluation.evaluator --config config.yaml
   File:    src/evaluation/evaluator.py
   Computes: ROC-AUC, PR-AUC, F1, confusion matrix, calibration
   Output:  evaluation/results/metrics.json, plots (PNG + HTML)

5. STREAMLIT DASHBOARD
   Command: streamlit run src/streamlit_app/app.py
   File:    src/streamlit_app/app.py
   Serves:  Multi-page web interface on localhost:8501

6. FASTAPI BACKEND
   Command: python -m uvicorn src.backend.app:app --host 0.0.0.0 --port 8000
   File:    src/backend/app.py
   Serves:  REST API on localhost:8000

7. NEXT.JS FRONTEND (built separately)
   Path:    frontend/ (see separate FRONTEND_SETUP.md)
   Runs:    npm run dev
   Connects to FastAPI backend


################################################################################
# CONFIGURATION HIERARCHY
################################################################################

The system uses a 3-level config override hierarchy:

1. config.yaml (base defaults) ← LOWEST priority
2. Command-line flags --lr 1e-5, --batch_size 128 ← MIDDLE
3. Environment variables GRAPH_LR=1e-5 ← HIGHEST priority

Example:
  python -m src.training.pretrain.main \\
    --config config.yaml \\
    --learning_rate 1e-5 \\
    --batch_size 256
  
  # Also respects env var: GRAPH_BATCH_SIZE=512


################################################################################
# DEPENDENCY GRAPH
################################################################################

config.yaml
    ↓
src/config_loader.py (parse YAML)
    ↓
src/utils.py (set seed, device, logging)
    ├→ src/data/ (load & preprocess)
    │   ├→ loaders/ (download datasets)
    │   └→ processors/ (normalize, split, sample)
    ├→ src/models/ (build models)
    │   ├→ foundation/ (GraphTransformer)
    │   ├→ baseline/ (GraphSAGE)
    │   └→ adapter/ (few-shot adapter)
    ├→ src/training/ (training loops)
    │   └→ pretrain/, meta_learning/, finetune/
    └→ src/evaluation/ (evaluation & plots)
        ├→ metrics/
        └→ visualization/


################################################################################
# NEXT PHASE
################################################################################

After PHASE 0 is validated, proceed with:

PHASE 1 → Data Pipeline
  - Implement: src/data/loaders/base_loader.py
  - Implement: src/data/loaders/pyg_loader.py
  - Implement: src/data/loaders/synthetic_loader.py
  - Implement: src/data/processors/*.py
