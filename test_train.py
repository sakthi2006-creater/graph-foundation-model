import traceback, sys
sys.stderr = sys.stdout
try:
    from src.training.pretrain.pretrain_model import PretrainTrainer
    t = PretrainTrainer()
    t.fit(num_epochs=1)
    print("PRETRAIN OK")
except Exception as e:
    traceback.print_exc()
    print("ERROR:", e)
