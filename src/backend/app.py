from fastapi import FastAPI
from pydantic import BaseModel
from src.config_loader import load_config
from src.models import GraphFoundationModel

app = FastAPI(title="Graph Link Prediction API")

config = load_config()

class PredictRequest(BaseModel):
    node_a: int
    node_b: int

@app.get("/")
def read_root():
    return {"message": "Graph Foundation Model API - ready!"}

@app.post("/predict")
def predict_link(request: PredictRequest):
    # Demo response
    prob = 0.87
    return {"node_a": request.node_a, "node_b": request.node_b, "probability": prob, "prediction": "Link exists"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

