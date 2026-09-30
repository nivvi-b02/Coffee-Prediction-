import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "coffee_processed.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "coffee_features.csv"
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nLoading cleaned data from:")
print(PROCESSED_FILE)

df = pd.read_csv(PROCESSED_FILE)

print(f"\nInput shape: {df.shape}")


# ============================================================
# EXPIRATION DATE
# ============================================================

# Convert values such as:
# "November 15th, 2023"
# into a proper datetime value.

df["Expiration"] = (
    df["Expiration"]
    .astype(str)
    .str.replace(
        r"(\d+)(st|nd|rd|th)",
        r"\1",
        regex=True
    )
)

df["Expiration"] = pd.to_datetime(
    df["Expiration"],
    errors="coerce"
)

# Extract useful date components

df["Expiration Year"] = (
    df["Expiration"].dt.year
)

df["Expiration Month"] = (
    df["Expiration"].dt.month
)

df["Expiration Day"] = (
    df["Expiration"].dt.day
)

# The original text/date column is no longer needed.

df = df.drop(
    columns=["Expiration"]
)


# ============================================================
# TOTAL PRODUCTION WEIGHT
# ============================================================

df["Total Production Weight"] = (
    df["Number of Bags"]
    * df["Bag Weight"]
)


# ============================================================
# DEFECT RATE
# ============================================================

df["Defect Rate"] = (
    df["Total Defects"]
    / df["Number of Bags"]
)


# ============================================================
# ALTITUDE CATEGORY
# ============================================================

def altitude_category(value):

    if pd.isna(value):
        return "Unknown"

    if value < 1000:
        return "Low"

    elif value < 1500:
        return "Medium"

    elif value < 2000:
        return "High"

    else:
        return "Very High"


df["Altitude Category"] = (
    df["Altitude"]
    .apply(altitude_category)
)


# ============================================================
# DISPLAY FEATURE SUMMARY
# ============================================================

print("\nNew features created:")

new_features = [
    "Expiration Year",
    "Expiration Month",
    "Expiration Day",
    "Total Production Weight",
    "Defect Rate",
    "Altitude Category",
]

print(new_features)


# ============================================================
# SAVE FEATURE-ENGINEERED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(f"Output shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print(OUTPUT_FILE)