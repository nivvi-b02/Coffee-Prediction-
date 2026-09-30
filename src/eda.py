import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

REPORTS_DIR = BASE_DIR / "reports"


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nLoading processed data from:")
print(PROCESSED_FILE)

df = pd.read_csv(PROCESSED_FILE)

print("\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# TARGET VARIABLE
# ============================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE: TOTAL CUP POINTS")
print("=" * 60)

print(df["Total Cup Points"].describe())


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_columns = [
    "Altitude",
    "Number of Bags",
    "Bag Weight",
    "Harvest Year",
    "Moisture Percentage",
    "Category One Defects",
    "Quakers",
    "Category Two Defects",
    "Total Defects",
    "Total Cup Points",
]

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Country of Origin",
    "Region",
    "In-Country Partner",
    "Variety",
    "Processing Method",
    "Color",
    "Certification Body",
]

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH TOTAL CUP POINTS")
print("=" * 60)

correlations = (
    df[numerical_columns]
    .corr()["Total Cup Points"]
    .sort_values(ascending=False)
)

print(correlations)


# ============================================================
# CREATE REPORTS DIRECTORY
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Total Cup Points"],
    kde=True
)

plt.title("Distribution of Total Cup Points")
plt.xlabel("Total Cup Points")
plt.ylabel("Number of Coffee Lots")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "target_distribution.png"
)

plt.close()


# ============================================================
# NUMERICAL CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df[numerical_columns].corr(),
    annot=True,
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "correlation_heatmap.png"
)

plt.close()


# ============================================================
# TARGET VS ALTITUDE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Altitude",
    y="Total Cup Points"
)

plt.title("Altitude vs Total Cup Points")
plt.xlabel("Altitude")
plt.ylabel("Total Cup Points")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "altitude_vs_quality.png"
)

plt.close()


# ============================================================
# TARGET VS TOTAL DEFECTS
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Total Defects",
    y="Total Cup Points"
)

plt.title("Total Defects vs Total Cup Points")
plt.xlabel("Total Defects")
plt.ylabel("Total Cup Points")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "defects_vs_quality.png"
)

plt.close()


# ============================================================
# PROCESSING METHOD VS QUALITY
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Processing Method",
    y="Total Cup Points"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.title("Processing Method vs Total Cup Points")
plt.xlabel("Processing Method")
plt.ylabel("Total Cup Points")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "processing_method_vs_quality.png"
)

plt.close()


# ============================================================
# COUNTRY VS QUALITY
# ============================================================

plt.figure(figsize=(10, 6))

country_order = (
    df.groupby("Country of Origin")["Total Cup Points"]
    .mean()
    .sort_values(ascending=False)
    .index
)

sns.boxplot(
    data=df,
    x="Country of Origin",
    y="Total Cup Points",
    order=country_order
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.title("Country of Origin vs Total Cup Points")
plt.xlabel("Country of Origin")
plt.ylabel("Total Cup Points")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "country_vs_quality.png"
)

plt.close()


# ============================================================
# EDA COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("\nEDA charts saved to:")
print(REPORTS_DIR)

print("\nGenerated files:")
print("- target_distribution.png")
print("- correlation_heatmap.png")
print("- altitude_vs_quality.png")
print("- defects_vs_quality.png")
print("- processing_method_vs_quality.png")
print("- country_vs_quality.png")
