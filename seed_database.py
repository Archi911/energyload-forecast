import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase
if not SUPABASE_URL or "your-project-id" in SUPABASE_URL:
    raise ValueError(" CRITICAL ERROR: You forgot to replace the placeholder URL in your .env file!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_block_to_supabase(block_path, mapping_path, batch_size=3000):
    print(" Loading datasets into migration engine...")
    
    if not os.path.exists(block_path):
        raise FileNotFoundError(f"Missing block file at: {block_path}. Please check your path!")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing mapping file at: {mapping_path}. Please check your path!")
        
    # Load data
    df = pd.read_csv(block_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # Standardize names and fix dirty data formats
    df = df.rename(columns={'tstp': 'timestamp', 'energy(kWh/hh)': 'energy'})
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df = df.dropna(subset=['energy'])
    
    # Format timestamp string for PostgreSQL
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Merge cluster mappings to identify swarms
    mapping_subset = mapping_df[['LCLid', 'cluster_id']]
    merged_df = pd.merge(df, mapping_subset, on='LCLid', how='inner')
    
    # Isolate columns matching our SQL schema exactly
    final_df = merged_df[['LCLid', 'timestamp', 'energy', 'cluster_id']]
    records = final_df.to_dict(orient='records')
    total_records = len(records)
    
    print(f" Prepared {total_records} rows for database seeding.")
    
    # Batch Insert Loop
    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table('household_telemetry').insert(batch).execute()
            print(f"   Streamed rows {i} to {min(i + batch_size, total_records)}...")
        except Exception as e:
            print(f"   Batch failed at row index {i}. Error: {e}")
            continue

if __name__ == "__main__":
    # Path to the cluster mapping file you already have in your data folder
    MAPPING_FILE = "data/household_cluster_mapping.csv"
    
    # 🛡️ Update this line to use our local test file!
    SAMPLE_BLOCK = "data/local_test_block.csv" 
    
    upload_block_to_supabase(SAMPLE_BLOCK, MAPPING_FILE)