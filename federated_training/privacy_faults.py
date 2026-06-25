import torch
import random
import copy

def apply_differential_privacy(local_weights, clip_threshold=1.0, noise_scale=0.01):
    """Secures client gradients against Model Inversion Attacks before sending to Cloud."""
    dp_weights = copy.deepcopy(local_weights)
    
    for key in dp_weights.keys():
        tensor = dp_weights[key].float()
        # Clip extreme values to prevent anomalous data from dominating
        tensor = torch.clamp(tensor, min=-clip_threshold, max=clip_threshold)
        # Inject Gaussian noise to mask specific household signatures
        noise = torch.randn_like(tensor) * noise_scale
        dp_weights[key] = tensor + noise
        
    return dp_weights

def simulate_edge_failures(active_clients, failure_rate=0.15):
    """Simulates real-world IoT network drops. Server must survive with remaining nodes."""
    surviving_clients = [c for c in active_clients if random.random() > failure_rate]
    dropped = len(active_clients) - len(surviving_clients)
    
    print(f" NETWORK FAULT: {dropped} edge nodes disconnected.")
    print(f" Proceeding with {len(surviving_clients)} surviving nodes...")
    
    return surviving_clients