import torch
import copy

class AdaptiveSwarmAggregator:
    def __init__(self, warmup_rounds=10):
        self.warmup_rounds = warmup_rounds

    def fed_avg(self, client_weights):
        """Fast baseline aggregation for early convergence."""
        global_weights = copy.deepcopy(client_weights[0])
        num_clients = len(client_weights)
        
        for key in global_weights.keys():
            for i in range(1, num_clients):
                global_weights[key] += client_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], num_clients)
        return global_weights

    def grey_wolf_optimization(self, client_weights, client_losses, current_iter, max_iter):
        """Heavy meta-heuristic for fine-tuning complex non-IID data."""
        num_clients = len(client_weights)
        sorted_indices = sorted(range(num_clients), key=lambda i: client_losses[i])
        
        alpha_w = client_weights[sorted_indices[0]]
        beta_w = client_weights[sorted_indices[1]] if num_clients > 1 else alpha_w
        delta_w = client_weights[sorted_indices[2]] if num_clients > 2 else alpha_w

        a = 2.0 - current_iter * (2.0 / max_iter)
        global_weights = copy.deepcopy(alpha_w)
        
        for key in global_weights.keys():
            device = global_weights[key].device
            tensor_shape = global_weights[key].shape
            
            # Wolf Pack Math (Simplified for core movement)
            r1, r2 = torch.rand(tensor_shape, device=device), torch.rand(tensor_shape, device=device)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            X1 = alpha_w[key] - A1 * torch.abs(C1 * alpha_w[key] - global_weights[key])

            r1, r2 = torch.rand(tensor_shape, device=device), torch.rand(tensor_shape, device=device)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            X2 = beta_w[key] - A2 * torch.abs(C2 * beta_w[key] - global_weights[key])

            r1, r2 = torch.rand(tensor_shape, device=device), torch.rand(tensor_shape, device=device)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            X3 = delta_w[key] - A3 * torch.abs(C3 * delta_w[key] - global_weights[key])

            global_weights[key] = (X1 + X2 + X3) / 3.0
            
        return global_weights

    def aggregate(self, round_num, total_rounds, client_weights, client_losses):
        """The Router: Decides which math to use based on the training phase."""
        if round_num <= self.warmup_rounds:
            print(f" Round {round_num}: Using FedAvg (Warmup)")
            return self.fed_avg(client_weights)
        else:
            print(f" Round {round_num}: Using GWO Swarm (Fine-Tuning)")
            return self.grey_wolf_optimization(client_weights, client_losses, round_num, total_rounds)