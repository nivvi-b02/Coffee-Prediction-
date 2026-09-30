import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_processed.csv"
)

MODELS_DIR = BASE_DIR / "models"


# ============================================================
# LOAD PROCESSED TRAINING DATA
# ============================================================

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("\nLoading processed training data from:")
print(TRAIN_FILE)

train_data = pd.read_csv(TRAIN_FILE)

print(f"\nTraining data shape: {train_data.shape}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_data.drop(
    columns=["Target"]
)

y_train = train_data["Target"]

print(f"\nFeatures: {X_train.shape}")
print(f"Target: {y_train.shape}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAIN MODELS
# ============================================================

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = model

    print("Training complete.")


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = MODELS_DIR / filename

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to:")
    print(model_path)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModels trained:")

for model_name in trained_models:
    print(f"- {model_name}")

print("\nSaved model files:")
print("- linear_regression.joblib")
print("- random_forest.joblib")
print("- gradient_boosting.joblib")