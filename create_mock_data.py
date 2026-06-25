import pandas as pd
import numpy as np

# Generating a small sample block matching your raw Kaggle half-hourly file structure exactly
mock_block = pd.DataFrame({
    'LCLid': ['MAC000002'] * 100 + ['MAC000003'] * 100,
    'tstp': pd.date_range(start="2012-10-12 00:30:00", periods=100, freq='30min').tolist() * 2,
    'energy(kWh/hh)': np.random.uniform(0.05, 1.5, 200).round(4)
})

# Save it locally
mock_block.to_csv("data/local_test_block.csv", index=False)
print(" Created a local 200-row test block at 'data/local_test_block.csv'!")