import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_processed.csv"
)

MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"

RESULTS_FILE = (
    REPORTS_DIR
    / "model_comparison.csv"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nLoading processed test data from:")
print(TEST_FILE)

test_data = pd.read_csv(TEST_FILE)

print(f"\nTest data shape: {test_data.shape}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = test_data.drop(
    columns=["Target"]
)

y_test = test_data["Target"]

print(f"\nFeatures: {X_test.shape}")
print(f"Target: {y_test.shape}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        MODELS_DIR / "linear_regression.joblib",

    "Random Forest":
        MODELS_DIR / "random_forest.joblib",

    "Gradient Boosting":
        MODELS_DIR / "gradient_boosting.joblib"
}


# ============================================================
# EVALUATE MODELS
# ============================================================

results = []


for model_name, model_path in models.items():

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    print("\nLoading model from:")
    print(model_path)

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nEvaluation Results:")

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2

    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# DISPLAY MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(RESULTS_FILE)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

print("\n" + "=" * 60)
print("BEST MODEL BY RMSE")
print("=" * 60)

print(f"\n{best_model}")