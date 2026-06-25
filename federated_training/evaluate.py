import torch
import numpy as np
import os
import pandas as pd
from lstm_seq2seq import VoltHiveLSTM
# Import your actual dataset loader here
from dataset_loader import load_cluster_data 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 48
FEATURES = 8

def load_swarm_model(swarm_id):
    path = f"models/best_model_swarm_{swarm_id}.pth"
    if not os.path.exists(path):
        return None
    model = VoltHiveLSTM(input_features=FEATURES, output_steps=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval() 
    return model

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    return mae, rmse

def evaluate_all_swarms():
    print("📊 Generating Research-Grade Evaluation Metrics...\n")
    
    total_mae = []
    total_rmse = []

    for swarm_id in range(3):
        model = load_swarm_model(swarm_id)
        if model is None: continue
        
        # Load the unseen test data for this specific cluster
        print(f"Loading test data for Swarm {swarm_id}...")
        # Note: Adjust the function call below to match how your dataset_loader works
        _, _, test_loader = load_cluster_data(cluster_id=swarm_id, batch_size=64)
        
        cluster_maes = []
        cluster_rmses = []
        
        with torch.no_grad():
            for sequences, actual_targets in test_loader:
                sequences = sequences.to(DEVICE)
                actual_targets = actual_targets.cpu().numpy()
                
                # Get predictions
                predictions = model(sequences).cpu().numpy()
                predictions = np.maximum(predictions, 0) # Zero out negative power
                
                # Calculate batch metrics
                mae, rmse = calculate_metrics(actual_targets, predictions)
                cluster_maes.append(mae)
                cluster_rmses.append(rmse)
        
        # Average metrics for this cluster
        final_cluster_mae = np.mean(cluster_maes)
        final_cluster_rmse = np.mean(cluster_rmses)
        
        total_mae.append(final_cluster_mae)
        total_rmse.append(final_cluster_rmse)
        
        print(f" Swarm {swarm_id} Results -> MAE: {final_cluster_mae:.4f} | RMSE: {final_cluster_rmse:.4f}\n")
    
    # Calculate the overall system average to compare with Table 9 in the paper
    print("==========================================")
    print(f" OVERALL SYSTEM AVERAGE")
    print(f"Overall MAE: {np.mean(total_mae):.4f}")
    print(f"Overall RMSE: {np.mean(total_rmse):.4f}")
    print("==========================================")

if __name__ == "__main__":
    evaluate_all_swarms()