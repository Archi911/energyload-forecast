import os
import sys
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException

# Cross-folder path resolution to find the neural network architecture
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'federated_training')))
from backend.lstm_seq2seq import VoltHiveLSTM
from backend.database import get_household_history, get_cluster_baseline

app = FastAPI(title="VoltHive Master API", version="3.0")

DEVICE = torch.device("cpu")
FEATURES = 8
SEQ_LEN = 48

# Memory states
swarm_experts = {}
cluster_map = {}
client_to_acorn_map = {}

@app.on_event("startup")
def initialize_system():
    print("Booting VoltHive MoE Inference Engine...")
    
    # Load Registry
    mapping_path = os.path.join(current_dir, "..", "data", "household_cluster_mapping.csv")
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        global cluster_map, client_to_acorn_map
        cluster_map = dict(zip(mapping_df['LCLid'], mapping_df['cluster_id']))
        client_to_acorn_map = dict(zip(mapping_df['LCLid'], mapping_df['Acorn']))
    
    # Load PyTorch Models
    for i in range(3):
        path = os.path.join(current_dir, "models", f"best_model_swarm_{i}.pth")
        if os.path.exists(path):
            model = VoltHiveLSTM(input_features=FEATURES, output_steps=SEQ_LEN).to(DEVICE)
            raw_state = torch.load(path, map_location=DEVICE)
            clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
            model.load_state_dict(clean_state)
            model.eval()
            swarm_experts[i] = model
            print(f"Swarm Expert {i} Online & Ready!")

def prepare_tensor_input(raw_db_data):
    """Transforms raw Supabase dictionaries into the rigid (1, 48, 8) LSTM Tensor."""
    matrix = []
    for row in raw_db_data:
        matrix.append([
            float(row.get('energy', 0.0)),
            int(row.get('hour', 0)),
            int(row.get('dayofweek', 0)),
            int(row.get('month', 1)),
            int(row.get('is_weekend', 0)),
            int(row.get('is_holiday', 0)),
            float(row.get('temperature', 15.0)),
            float(row.get('humidity', 0.5))
        ])
    return torch.tensor([matrix], dtype=torch.float32).to(DEVICE)

@app.get("/predict/household/{lcl_id}")
async def forecast_household(lcl_id: str):
    """Predicts load for a single household using ITS OWN exact data."""
    if lcl_id not in cluster_map:
        raise HTTPException(status_code=404, detail="Meter ID not found in registry.")
    
    cluster_id = cluster_map[lcl_id]
    db_history = get_household_history(lcl_id)
    
    if not db_history or len(db_history) < 48:
        raise HTTPException(status_code=500, detail="Not enough historical data for this node.")
        
    input_tensor = prepare_tensor_input(db_history)
    with torch.no_grad():
        raw_pred = swarm_experts[cluster_id](input_tensor).squeeze().tolist()
        clean_forecast = [max(0.0, float(val)) for val in raw_pred]
        
    return {
        "lcl_id": lcl_id,
        "cluster_id": int(cluster_id),
        "acorn_group": client_to_acorn_map.get(lcl_id, "Unknown"),
        "forecast_48_steps": clean_forecast,
        "current_load": float(input_tensor[0, -1, 0].item()),
        "peak_load": round(max(clean_forecast), 4)
    }

@app.get("/predict/cluster/{cluster_id}")
async def forecast_cluster_aggregation(cluster_id: int):
    """Predicts aggregate grid capacity."""
    db_history = get_cluster_baseline(cluster_id)
    if not db_history:
        raise HTTPException(status_code=500, detail="Database empty.")
        
    input_tensor = prepare_tensor_input(db_history)
    
    with torch.no_grad():
        raw_pred = swarm_experts[cluster_id](input_tensor).squeeze().tolist()
        clean_forecast = [max(0.0, float(val)) for val in raw_pred]
        
    total_cluster_nodes = sum(1 for cid in cluster_map.values() if cid == cluster_id)
    aggregated_sum_curve = [val * total_cluster_nodes for val in clean_forecast]
    
    return {
        "cluster_id": cluster_id,
        "total_nodes": total_cluster_nodes,
        "aggregated_tomorrow_kwh": round(sum(aggregated_sum_curve), 2),
        "aggregated_curve": aggregated_sum_curve
    }