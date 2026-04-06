# Graph Foundation Model - Execution Tracker

Current Working Directory: c:/Users/USER/Desktop/santha

## PHASE A ✅ Environment Setup

- [x] Create virtual environment
- [x] Activate venv
- [x] Upgrade pip
- [x] Install requirements.txt
- [x] Verify PyTorch + PyG

## PHASE B 🔄 GPU/Device Check

- [ ] Check torch.cuda.is_available()
- [ ] Log GPU name/CUDA version
- [ ] Set config.device accordingly

## PHASE C ✅ Data Pipeline

- [x] Run tests/test_phase_1_data_pipeline.py
- [x] Validate Cora/PubMed/etc. (or synthetic fallback)
- [x] Check data/cache/ populated

## PHASE D 🚀 Backend (Priority)

- [ ] uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
- [ ] Test /health, /predict endpoints

## PHASE E 🎨 Streamlit Dashboard

- [ ] streamlit run src/streamlit_app/app.py
- [ ] Verify localhost:8501 all 3 tabs

## PHASE F 🧪 Full ML Pipeline

- [ ] Quick pretrain: src.training.pretrain.main --num_epochs 1
- [ ] Evaluation: src.evaluation.evaluator

## VSCode Extensions (Recommended)

```
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)
- Ruff (charliermarsh.ruff)
```

**Next:** PHASE A pip install → PHASE B GPU check → PHASE C data test → **PHASE D FastAPI backend first**.
