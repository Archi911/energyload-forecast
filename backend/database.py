import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Load Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

def get_last_24h_from_supabase(cluster_id: int):
    """
    Fetches the most recent 48 telemetry readings (8 features) for a specific cluster.
    """
    if not supabase:
        raise ValueError("Supabase credentials missing!")
    
    try:
        # Fetching all 8 features directly (Denormalized Feature Store Pattern)
        response = supabase.table('household_telemetry') \
            .select('energy, hour, dayofweek, month, is_weekend, is_holiday, temperature, humidity') \
            .eq('cluster_id', cluster_id) \
            .order('timestamp', desc=True) \
            .limit(48) \
            .execute()
        
        data = response.data
        if not data or len(data) < 48:
            print(f"Warning: Insufficient data for cluster {cluster_id}")
            return None
        
        # Reverse to chronological order (oldest -> newest) for the LSTM
        return data[::-1]
        
    except Exception as e:
        print(f"Database Query Failed: {e}")
        return None

def get_household_history(lcl_id: str):
    """Fetches the exact 48-step time series for ONE specific household."""
    if not supabase: return None
    try:
        response = supabase.table('household_telemetry') \
            .select('energy, hour, dayofweek, month, is_weekend, is_holiday, temperature, humidity') \
            .eq('LCLid', lcl_id) \
            .order('timestamp', desc=True) \
            .limit(48) \
            .execute()
        return response.data[::-1] if response.data else None
    except Exception as e:
        print(f"Household Query Failed: {e}")
        return None

def get_cluster_baseline(cluster_id: int):
    """Fetches a clean 48-step baseline shape for the cluster aggregate."""
    if not supabase: return None
    try:
        response = supabase.table('household_telemetry') \
            .select('energy, hour, dayofweek, month, is_weekend, is_holiday, temperature, humidity') \
            .eq('cluster_id', cluster_id) \
            .order('timestamp', desc=True) \
            .limit(48) \
            .execute()
        return response.data[::-1] if response.data else None
    except Exception as e:
        print(f"Cluster Query Failed: {e}")
        return None