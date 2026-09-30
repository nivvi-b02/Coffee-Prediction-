import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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


# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

print("=" * 60)
print("PREPROCESSING")
print("=" * 60)

print("\nLoading feature-engineered data from:")
print(FEATURE_FILE)

df = pd.read_csv(FEATURE_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# DEFINE TARGET
# ============================================================

TARGET = "Total Cup Points"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nTarget variable:")
print(TARGET)

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_columns)

print("\nCategorical features:")
print(categorical_columns)


# ============================================================
# NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)


# ============================================================
# COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        ),
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ============================================================
# FIT PREPROCESSOR ON TRAINING DATA
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("Preprocessing complete.")


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

train_data = pd.DataFrame(
    X_train_processed
)

train_data["Target"] = y_train.reset_index(drop=True)

test_data = pd.DataFrame(
    X_test_processed
)

test_data["Target"] = y_test.reset_index(drop=True)

train_output = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_processed.csv"
)

test_output = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_processed.csv"
)

train_data.to_csv(
    train_output,
    index=False
)

test_data.to_csv(
    test_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed testing shape: {X_test_processed.shape}")

print("\nSaved files:")
print(train_output)
print(test_output)
print(PREPROCESSOR_FILE)