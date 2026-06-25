import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import pandas as pd
import os
import glob
import re
import gc 

# --- IMPORT CUSTOM MODULES ---
from lstm_seq2seq import VoltHiveLSTM
from swarm_aggregator import AdaptiveSwarmAggregator
from privacy_faults import apply_differential_privacy, simulate_edge_failures
from dataset_loader import create_client_dataloaders

# =====================================================================
# 1. HYPERPARAMETERS & CONFIGURATION (UPDATED FOR SWARM 0)
# =====================================================================
FL_ROUNDS = 40          # Increased for deeper, smoother convergence
WARMUP_ROUNDS = 10      
EPOCHS_PER_CLIENT = 1   # Reduced to 1 to prevent client drift & slash round times
CLIENT_FRACTION = 0.2   # Samples 20% of nodes per round (~56 households)
LEARNING_RATE = 0.0001  # Reduced by 10x for careful, precise weight updates
DP_CLIP = 1.0          
DP_NOISE = 0.01        
PATIENCE = 10           # Increased leash to survive temporary validation plateaus

TARGET_SWARM = 1        # Switched target to train the second dataset

# =====================================================================
# 2. SETUP & INITIALIZATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Initializing VoltHive Swarm on: {device}")
if torch.cuda.device_count() > 1:
    print(f" Multi-GPU Detected! Utilizing {torch.cuda.device_count()} GPUs.")

global_model = VoltHiveLSTM(input_features=8, output_steps=48).to(device)
if torch.cuda.device_count() > 1:
    global_model = nn.DataParallel(global_model) 
global_weights = copy.deepcopy(global_model.state_dict())

aggregator = AdaptiveSwarmAggregator(warmup_rounds=WARMUP_ROUNDS)
training_logs = {"round": [], "aggregator": [], "val_loss": [], "active_nodes": []}
os.makedirs("checkpoints", exist_ok=True)

best_val_loss = float('inf')
patience_counter = 0
start_round = 1

checkpoint_files = glob.glob(f"checkpoints/global_model_swarm_{TARGET_SWARM}_round_*.pth")
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    match = re.search(r'round_(\d+)', latest_checkpoint)
    if match:
        start_round = int(match.group(1)) + 1
        global_weights = torch.load(latest_checkpoint)
        global_model.load_state_dict(global_weights)
        print(f" CRASH RECOVERY: Resuming Swarm {TARGET_SWARM} from Round {start_round}")

# =====================================================================
# 3. AUTO-DISCOVERY & DISK STREAMING SETUP
# =====================================================================
print("\n Scanning Kaggle File System for Data...")

DATA_DIR = ""
for root, dirs, files in os.walk('/kaggle/input'):
    if 'informations_households.csv' in files:
        DATA_DIR = root
        break

if not DATA_DIR:
    raise FileNotFoundError("CRITICAL ERROR: Could not find the Smart Meters dataset!")

mapping_df = pd.read_csv("household_cluster_mapping.csv") 
swarm_clients = mapping_df[mapping_df['cluster_id'] == TARGET_SWARM]['LCLid'].unique().tolist()
all_client_ids = swarm_clients
print(f" Isolated Swarm {TARGET_SWARM}. Found {len(all_client_ids)} households.")

weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather_hourly_darksky.csv"))
holidays_df = pd.read_csv(os.path.join(DATA_DIR, "uk_bank_holidays.csv"))

info_df = pd.read_csv(os.path.join(DATA_DIR, "informations_households.csv"))
client_to_block_map = dict(zip(info_df['LCLid'], info_df['file']))
print(" Streaming Index Built. Ready for Out-of-Core Processing.")

print("🗄️ Indexing physical block paths...")
block_paths = {}
for root, dirs, files in os.walk('/kaggle/input'):
    if 'halfhourly' in root.lower():  
        for file in files:
            if file.startswith('block_') and file.endswith('.csv'):
                block_paths[file] = os.path.join(root, file)

# =====================================================================
# 4. THE FEDERATED LEARNING LOOP
# =====================================================================
for round_num in range(start_round, FL_ROUNDS + 1):
    print(f"\n{'='*50}\n FEDERATED ROUND {round_num}/{FL_ROUNDS}\n{'='*50}")
    
    # Randomly select a fraction of households to speed up round times
    sampled_clients = random.sample(all_client_ids, max(1, int(len(all_client_ids) * CLIENT_FRACTION)))
    active_clients = simulate_edge_failures(sampled_clients, failure_rate=0.15)
    
    if not active_clients:
        print(" All sampled nodes disconnected. Skipping round.")
        continue

    client_updates = []
    client_losses = []
    
    # ---  LOCAL EDGE TRAINING & PERFECT PREPROCESSING ---
    for client_id in active_clients:
        block_filename = client_to_block_map.get(client_id)
        
        if pd.isna(block_filename):
            print(f" Node {client_id}: Missing from index. Skipping.")
            continue
            
        block_filename = str(block_filename)
        if not block_filename.endswith('.csv'):
            block_filename += '.csv'
            
        if block_filename not in block_paths:
            print(f" Node {client_id}: {block_filename} not found on disk. Skipping.")
            continue
            
        # 1. Lazy Load precisely one block
        block_path = block_paths[block_filename]
        temp_block_df = pd.read_csv(block_path, low_memory=False)
            
        raw_df = temp_block_df[temp_block_df['LCLid'] == client_id].copy()
        
        del temp_block_df
        gc.collect() 
        
        if raw_df.empty:
            print(f" Node {client_id}: No telemetry data found inside block. Skipping.")
            continue
            
        raw_df.columns = raw_df.columns.str.strip()
        
        rename_map = {
            'tstp': 'timestamp', 
            'energy(kWh/hh)': 'energy',
            'energy(kwh/hh)': 'energy',
            'Energy': 'energy'
        }
        raw_df = raw_df.rename(columns=rename_map)
        
        if 'energy' not in raw_df.columns or 'timestamp' not in raw_df.columns:
            print(f" Node {client_id}: Unexpected columns {list(raw_df.columns)}. Skipping.")
            continue
            
        raw_df['energy'] = pd.to_numeric(raw_df['energy'], errors='coerce')
        raw_df = raw_df.dropna(subset=['energy']) 
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
            
        train_loader, val_loader, test_loader = create_client_dataloaders(raw_df, weather_df, holidays_df)
        if train_loader is None or val_loader is None:
            print(f"  Node {client_id}: Preprocessing failed (Insufficient data). Skipping.")
            continue
            
        local_model = VoltHiveLSTM(input_features=8, output_steps=48).to(device)
        if torch.cuda.device_count() > 1:
             local_model = nn.DataParallel(local_model)
        local_model.load_state_dict(global_weights) 
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
        
        local_model.train()
        for epoch in range(EPOCHS_PER_CLIENT):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = local_model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
            
        local_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = local_model(batch_x)
                val_loss += criterion(predictions, batch_y).item()
        
        final_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        secured_weights = apply_differential_privacy(local_model.state_dict(), DP_CLIP, DP_NOISE)
        client_updates.append(secured_weights)
        client_losses.append(final_val_loss)
        
        print(f" Node {client_id} successfully trained & secured! (Loss: {final_val_loss:.4f})")

    # --- CLOUD AGGREGATION ---
    if not client_updates:
        print(" No valid edge models returned this round.")
        continue

    print(f"\n Server Aggregating {len(client_updates)} Secured Edge Models...")
    global_weights = aggregator.aggregate(round_num, FL_ROUNDS, client_updates, client_losses)
    avg_round_loss = sum(client_losses) / len(client_losses)
    current_algo = "FedAvg" if round_num <= WARMUP_ROUNDS else "GWO"
    
    training_logs["round"].append(round_num)
    training_logs["aggregator"].append(current_algo)
    training_logs["val_loss"].append(avg_round_loss)
    training_logs["active_nodes"].append(len(client_updates))
    
    print(f" Round {round_num} Complete | Algo: {current_algo} | Global Avg Loss: {avg_round_loss:.4f}")
    
    if round_num % 5 == 0 or round_num == FL_ROUNDS:
        torch.save(global_weights, f"checkpoints/global_model_swarm_{TARGET_SWARM}_round_{round_num}.pth")

    if avg_round_loss < best_val_loss:
        best_val_loss = avg_round_loss
        patience_counter = 0
        torch.save(global_weights, f"checkpoints/best_model_swarm_{TARGET_SWARM}.pth")
        print(f" New Best Model Locked In! Saved to checkpoints/best_model_swarm_{TARGET_SWARM}.pth")
    else:
        patience_counter += 1
        print(f"  Validation loss plateau. Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\n EARLY STOPPING: Swarm {TARGET_SWARM} has mathematically converged at Round {round_num}.")
        break 

print("\n VoltHive Architecture Training Complete!")
pd.DataFrame(training_logs).to_csv(f"volthive_metrics_swarm_{TARGET_SWARM}.csv", index=False)