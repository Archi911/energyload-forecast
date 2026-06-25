import torch
import torch.nn as nn

class VoltHiveLSTM(nn.Module):
    def __init__(self, input_features=8, hidden_size=64, num_layers=2, output_steps=48, dropout=0.1):
        super(VoltHiveLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_steps)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Extract the final memory state of the past 24 hours
        final_time_step = lstm_out[:, -1, :]
        
        out = self.relu(self.fc1(final_time_step))
        out = self.dropout(out)
        return self.fc2(out) # Returns (Batch, 48) future loads