import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class SmartGridDataset(Dataset):
    def __init__(self, data_matrix, seq_length=48, pred_horizon=48):
        self.data = data_matrix
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon

    def __len__(self):
        # Ensure we don't go out of bounds when predicting 48 steps into the future
        return max(0, len(self.data) - self.seq_length - self.pred_horizon)

    def __getitem__(self, index):
        # X: The past 48 steps (all 8 features)
        x = self.data[index : index + self.seq_length, :]
        # Y: The next 48 steps (only the 'energy' column, which is index 0)
        y = self.data[index + self.seq_length : index + self.seq_length + self.pred_horizon, 0]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_client_dataloaders(raw_df, weather_df, holidays_df, seq_length=48, pred_horizon=48, batch_size=32):
    """
    Takes the raw telemetry, merges external covariates, extracts time features,
    and builds the PyTorch DataLoaders.
    """
    df = raw_df.copy()
    
    # ---------------------------------------------------------
    # 1. TIME FEATURE ENGINEERING
    # ---------------------------------------------------------
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['date'] = df['timestamp'].dt.normalize() # Strip time for daily merges

    # ---------------------------------------------------------
    # 2. HOLIDAY MERGE
    # ---------------------------------------------------------
    # Find the correct column name dynamically
    hol_col = 'Bank holidays' if 'Bank holidays' in holidays_df.columns else holidays_df.columns[0]
    holidays_df['date'] = pd.to_datetime(holidays_df[hol_col], format='%Y-%m-%d', errors='coerce')
    holidays_df['is_holiday'] = 1
    
    df = pd.merge(df, holidays_df[['date', 'is_holiday']], on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

    # ---------------------------------------------------------
    # 3. WEATHER MERGE
    # ---------------------------------------------------------
    weather_df['weather_time'] = pd.to_datetime(weather_df['time'])
    
    # Extract temperature and humidity. Provide safe fallbacks if the dataset is missing them.
    temp_col = 'temperature' if 'temperature' in weather_df.columns else 'temperatureMax'
    humid_col = 'humidity' if 'humidity' in weather_df.columns else 'visibility' 
    
    available_weather = ['weather_time']
    if temp_col in weather_df.columns: available_weather.append(temp_col)
    if humid_col in weather_df.columns: available_weather.append(humid_col)
        
    temp_weather = weather_df[available_weather].rename(columns={'weather_time': 'hour_timestamp'})
    
    # Round the 30-min telemetry to the nearest hour to match the hourly weather
    df['hour_timestamp'] = df['timestamp'].dt.floor('h')
    df = pd.merge(df, temp_weather, on='hour_timestamp', how='left')
    
    # Forward fill any missing weather data, fallback to averages if completely missing
    if temp_col in df.columns:
        df['temperature'] = df[temp_col].ffill().bfill().fillna(15.0)
    else:
        df['temperature'] = 15.0 # Dummy baseline
        
    if humid_col in df.columns:
        df['humidity'] = df[humid_col].ffill().bfill().fillna(0.5)
    else:
        df['humidity'] = 0.5 # Dummy baseline

    # ---------------------------------------------------------
    # 4. FINAL MATRIX ASSEMBLY
    # ---------------------------------------------------------
    # MUST MATCH input_features=8 IN YOUR LSTM EXACTLY
    features = ['energy', 'hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday', 'temperature', 'humidity']
    
    # Drop rows with broken NaN energy values
    df = df.dropna(subset=features)
    
    # Safety Check: Does this house have enough data to train on?
    if len(df) < (seq_length + pred_horizon + 10):
        return None, None, None

    # Convert to pure mathematical tensor
    data_matrix = df[features].values.astype(np.float32)

    # ---------------------------------------------------------
    # 5. PYTORCH DATALOADERS (70% Train, 20% Val, 10% Test)
    # ---------------------------------------------------------
    n = len(data_matrix)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    train_data = data_matrix[:train_end]
    val_data = data_matrix[train_end:val_end]
    test_data = data_matrix[val_end:]
    
    train_loader = DataLoader(SmartGridDataset(train_data, seq_length, pred_horizon), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(SmartGridDataset(val_data, seq_length, pred_horizon), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(SmartGridDataset(test_data, seq_length, pred_horizon), batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, test_loader