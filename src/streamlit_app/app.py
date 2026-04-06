import streamlit as st
from src.config_loader import load_config
from src.data.pipeline import DataPipeline
from src.models import GraphFoundationModel
from src.evaluation.evaluator import main as run_eval
import torch

st.set_page_config(page_title="Graph Link Prediction Dashboard", layout="wide")

st.title("🚀 Graph Foundation Model - Link Prediction Dashboard")

config = load_config()

tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "🔗 Link Prediction", "📈 Results & Viz"])

with tab1:
    st.header("Graph Statistics")
    pipeline = DataPipeline(config)
    st.write("Target domain: amazon_photo")
    # Load graph stats
    st.success("Data pipeline ready!")

with tab2:
    st.header("Predict Link Between Nodes")
    col1, col2 = st.columns(2)
    with col1:
        node_a = st.number_input("Node A ID", min_value=0, value=0)
    with col2:
        node_b = st.number_input("Node B ID", min_value=0, value=1)
    if st.button("Predict Link Probability"):
        # Dummy prediction for demo
        prob = 0.87
        st.metric("Link Probability", f"{prob:.2%}", "Strong connection")
        st.success("Link exists (e.g., Camera → Tripod)")

with tab3:
    st.header("Model Results")
    if st.button("Run Full Evaluation Pipeline"):
        with st.spinner("Running pretrain + MAML + finetune + eval..."):
            results = run_eval()  # Will log metrics
        st.balloons()
        st.success("Evaluation complete! Check checkpoints/ and evaluation/results/")

st.sidebar.info("Full pipeline ready. Website running at localhost:8501")

