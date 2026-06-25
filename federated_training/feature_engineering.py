import pandas as pd
import numpy as np

def detect_and_fix_sensor_faults(df, column='energy'):
    """
    Scans for hardware malfunctions (impossible energy spikes) using IQR.
    Replaces faults with NaN, which are then fixed via interpolation.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define an "Impossible Spike" (e.g., 4x the IQR above the 75th percentile)
    upper_bound = Q3 + 4 * IQR
    lower_bound = 0  # Energy cannot be negative

    faults = (df[column] > upper_bound) | (df[column] < lower_bound)
    
    if faults.sum() > 0:
        # Quarantine the bad data
        df.loc[faults, column] = np.nan
        
    return df

def enrich_and_interpolate(raw_df, weather_df, holidays_df):
    """Merges covariates and mathematically bridges missing sensor gaps."""
    # 1. Ensure absolute chronological order
    df = raw_df.sort_values(['LCLid', 'timestamp']).reset_index(drop=True)
    
    # 2. Clean text errors
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    
    # 3. STATISTICAL FAULT DETECTOR
    df = detect_and_fix_sensor_faults(df, column='energy')
    
    # 4. Interpolate (Fixes both missing offline data and the quarantined faults)
    df['energy'] = df['energy'].interpolate(method='linear', limit_direction='both')
    
    # 5. Inject Weather Covariates
    weather_df['day'] = pd.to_datetime(weather_df['time']).dt.normalize()
    df = pd.merge(df, weather_df[['day', 'temperatureMax', 'cloudCover']], on='day', how='left')
    df[['temperatureMax', 'cloudCover']] = df[['temperatureMax', 'cloudCover']].ffill().bfill()
    
    # 6. Inject UK Bank Holidays
    holidays_df['day'] = pd.to_datetime(holidays_df['Bank holidays'], format='%Y-%m-%d')
    holidays_df['is_holiday'] = 1
    df = pd.merge(df, holidays_df[['day', 'is_holiday']], on='day', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    
    # 7. Extract Temporal Cyclical Features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    return df