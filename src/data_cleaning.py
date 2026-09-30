import pandas as pd
from pathlib import Path

from data_loading import load_raw_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "coffee_processed.csv"


# ============================================================
# ALTITUDE
# ============================================================

def convert_altitude(value):
    """
    Convert altitude values into a numeric midpoint.

    Examples:
    1700       -> 1700
    1700-1930  -> 1815
    150-250    -> 200
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Handle ranges with hyphen
    if "-" in value:

        parts = value.replace(" ", "").split("-")

        try:
            low = float(parts[0])
            high = float(parts[1])

            return (low + high) / 2

        except (ValueError, IndexError):
            return None

    # Handle single values
    try:
        return float(value)

    except ValueError:
        return None


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print(f"Input shape: {df.shape}")


    # --------------------------------------------------------
    # REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------------

    drop_columns = [
        "Unnamed: 0",
        "ID",
        "Lot Number",
        "ICO Number",
        "Farm Name",
        "Producer",
        "Company",
        "Mill",
        "Owner",
        "Certification Address",
        "Certification Contact",
        "Status",

        # Sensory evaluation scores
        # These are excluded to prevent target leakage.
        "Aroma",
        "Flavor",
        "Aftertaste",
        "Acidity",
        "Body",
        "Balance",
        "Uniformity",
        "Clean Cup",
        "Sweetness",
        "Overall",
        "Defects",
    ]

    df = df.drop(
        columns=drop_columns,
        errors="ignore"
    )


    # --------------------------------------------------------
    # ALTITUDE
    # --------------------------------------------------------

    df["Altitude"] = df["Altitude"].apply(
        convert_altitude
    )


    # --------------------------------------------------------
    # BAG WEIGHT
    # --------------------------------------------------------

    df["Bag Weight"] = (
        df["Bag Weight"]
        .astype(str)
        .str.replace(" kg", "", regex=False)
    )

    df["Bag Weight"] = pd.to_numeric(
        df["Bag Weight"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # HARVEST YEAR
    # --------------------------------------------------------

    df["Harvest Year"] = (
        df["Harvest Year"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    df["Harvest Year"] = pd.to_numeric(
        df["Harvest Year"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # GRADING DATE
    # --------------------------------------------------------

    # Grading Date contains too many missing/invalid values
    # in this dataset, so it is excluded.

    df = df.drop(
        columns=["Grading Date"],
        errors="ignore"
    )


    # --------------------------------------------------------
    # DEFECT FEATURES
    # --------------------------------------------------------

    df["Total Defects"] = (
        df["Category One Defects"]
        + df["Category Two Defects"]
    )


    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "Country of Origin",
        "Region",
        "Variety",
        "Processing Method",
        "Color",
        "In-Country Partner",
        "Certification Body",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)

    print(f"Processed shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSaved to:")
    print(OUTPUT_FILE)

    return df


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    # Load raw data using the data loading module
    df = load_raw_data()

    # Clean the loaded data
    clean_data(df)