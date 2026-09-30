import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "df_arabica_clean.csv"


# ============================================================
# DATA INGESTION
# ============================================================

def load_raw_data():

    print("=" * 60)
    print("DATA INGESTION")
    print("=" * 60)

    print(f"Loading dataset from:\n{RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    print("\nDataset loaded successfully.")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# TEST DATA LOADING
# ============================================================

if __name__ == "__main__":

    df = load_raw_data()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())