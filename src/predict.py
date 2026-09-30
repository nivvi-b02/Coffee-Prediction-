import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "coffee_features.csv"
)

MODELS_DIR = BASE_DIR / "models"

PREPROCESSOR_FILE = (
    MODELS_DIR
    / "preprocessor.joblib"
)

MODEL_FILE = (
    MODELS_DIR
    / "random_forest.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("COFFEE QUALITY PREDICTION")
print("=" * 60)

print("\nLoading feature-engineered data from:")
print(FEATURE_FILE)

df = pd.read_csv(FEATURE_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# DEFINE TARGET
# ============================================================

TARGET = "Total Cup Points"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("\n" + "=" * 60)
print("LOADING PREPROCESSOR")
print("=" * 60)

print("\nLoading preprocessor from:")
print(PREPROCESSOR_FILE)

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)

print("Preprocessor loaded successfully.")


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING RANDOM FOREST MODEL")
print("=" * 60)

print("\nLoading model from:")
print(MODEL_FILE)

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# SELECT ONE COFFEE RECORD
# ============================================================

# Use the first record from the dataset
# as a test prediction.

sample = X.iloc[[0]]

actual_score = y.iloc[0]


# ============================================================
# PREPROCESS SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING SAMPLE")
print("=" * 60)

sample_processed = preprocessor.transform(
    sample
)

print("Sample preprocessing complete.")


# ============================================================
# MAKE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("MAKING PREDICTION")
print("=" * 60)

prediction = model.predict(
    sample_processed
)

predicted_score = prediction[0]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"\nPredicted Coffee Quality Score: {predicted_score:.2f}")

print(f"Actual Coffee Quality Score:    {actual_score:.2f}")

print(
    f"Prediction Error:              "
    f"{abs(predicted_score - actual_score):.2f}"
)


# ============================================================
# QUALITY INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("QUALITY INTERPRETATION")
print("=" * 60)

if predicted_score >= 85:
    quality = "High Quality"

elif predicted_score >= 82:
    quality = "Good Quality"

else:
    quality = "Lower Quality"


print(f"\nPredicted Quality Category: {quality}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)
