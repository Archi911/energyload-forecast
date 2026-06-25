# VoltHive: Decentralized Energy Load Forecasting

A high-performance, full-stack machine learning architecture designed for decentralized smart meter load forecasting using a Mixture-of-Experts (MoE) LSTM approach.

## 🚀 Live Demo
Access the interactive dashboard here: **[VoltHive Master Substation Console](https://huggingface.co/spaces/artch23/volthive-forecasting)**

##  Architecture Overview
VoltHive implements an efficient MLOps pipeline to bridge heavy neural network inference with lightweight, real-time visualization.

- **Frontend:** Streamlit dashboard for real-time heatmap visualization and node-level energy capacity analysis.
- **Backend:** FastAPI-based inference engine optimized for low-latency delivery.
- **Model:** Mixture-of-Experts (MoE) LSTM architecture handling 4,400+ smart meters with 7.9% NRMSE.
- **Data & Ops:** Supabase (PostgreSQL) for telemetry storage, containerized with Docker for seamless deployment on Hugging Face Spaces.


##  Tech Stack
- **Languages:** Python
- **ML/DL:** PyTorch, Pandas, Scikit-learn
- **API/Web:** FastAPI, Uvicorn, Streamlit, Plotly
- **Infrastructure:** Docker, Hugging Face Spaces, Supabase

## 📂 Project Structure
```text
├── backend/          # FastAPI inference API & model logic
├── frontend/         # Streamlit dashboard UI
├── federated_training/# Model training & aggregation scripts
├── Dockerfile        # Container configuration
└── requirements.txt  # Project dependencies
