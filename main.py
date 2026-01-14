import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_engineering import create_features
from model_training import build_pipelines, get_param_grids, train_models
from evaluation import evaluate_model
from extrema_detection import find_extrema, find_midpoints
from interpretability import generate_interpretability_outputs


# -----------------------------
# Configuration
# -----------------------------
ISLAND_NAME = "Fuerteventura"
SPEI_SCALE = "SPEI6"

DATA_PATH = "data/Date_fuerteventura_SPEI6.csv"   # not tracked in Git
OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explanations")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)


# -----------------------------
# 1. Load data
# -----------------------------
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
data.set_index("Date", inplace=True)

end_date = data.index[-1]
start_date = end_date - timedelta(days=365.25 * 10)


# -----------------------------
# 2. Feature engineering
# -----------------------------
data = create_features(data)

X = data.drop("Average", axis=1)
y = data["Average"]

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


# -----------------------------
# 3. Model training
# -----------------------------
rf_pipeline, xgb_pipeline = build_pipelines()
rf_param_grid, xgb_param_grid = get_param_grids()

rf_random, xgb_random = train_models(
    X_train, y_train,
    rf_pipeline, xgb_pipeline,
    rf_param_grid, xgb_param_grid
)

rf_model = rf_random.best_estimator_
xgb_model = xgb_random.best_estimator_


# -----------------------------
# 4. Predictions and export
# -----------------------------
rf_pred_full = rf_model.predict(X)
xgb_pred_full = xgb_model.predict(X)

comparison_df = pd.DataFrame({
    "Date": data.index,
    f"Actual {SPEI_SCALE}": y,
    "Random Forest Prediction": rf_pred_full,
    "XGBoost Prediction": xgb_pred_full
})

comparison_df.to_csv(
    os.path.join(TABLES_DIR, f"{SPEI_SCALE.lower()}_actual_vs_predictions_full_{ISLAND_NAME.lower()}.csv"),
    index=False
)


# -----------------------------
# 5. Evaluation
# -----------------------------
results_full = {
    "Random Forest": evaluate_model(rf_random, X, y),
    "XGBoost": evaluate_model(xgb_random, X, y)
}

results_train = {
    "Random Forest": evaluate_model(rf_random, X_train, y_train),
    "XGBoost": evaluate_model(xgb_random, X_train, y_train)
}

results_test = {
    "Random Forest": evaluate_model(rf_random, X_test, y_test),
    "XGBoost": evaluate_model(xgb_random, X_test, y_test)
}

print("Model performance on full dataset:")
for name, metrics in results_full.items():
    print(f"{name}: {metrics}")

print("\nModel performance on training set:")
for name, metrics in results_train.items():
    print(f"{name}: {metrics}")

print("\nModel performance on test set:")
for name, metrics in results_test.items():
    print(f"{name}: {metrics}")


# -----------------------------
# 6. Plot: train/test predictions
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(data.index, y, label=f"Actual {SPEI_SCALE}", color="black", linewidth=2)

for name, model, color in [
    ("Random Forest", rf_model, "red"),
    ("XGBoost", xgb_model, "blue")
]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    plt.plot(X_train.index, train_pred,
             label=f"{name} (Train Fit)",
             linestyle="--", alpha=0.7, color=color)

    plt.plot(X_test.index, test_pred,
             label=f"{name} (Test Prediction)",
             linestyle="-", linewidth=2, color=color)

plt.title(f"{SPEI_SCALE} Time Series for {ISLAND_NAME}: Actual vs Model Predictions", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel(SPEI_SCALE, fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{SPEI_SCALE.lower()}_train_test_predictions_{ISLAND_NAME.lower()}.png"), dpi=300)
plt.close()


# -----------------------------
# 7. Plot: full predictions
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label=f"Actual {SPEI_SCALE}")
plt.plot(data.index, rf_pred_full, label="Random Forest Prediction")
plt.plot(data.index, xgb_pred_full, label="XGBoost Prediction")
plt.title(f"{SPEI_SCALE} Time Series and Predictions ({ISLAND_NAME})", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel(SPEI_SCALE, fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"{SPEI_SCALE.lower()}_predictions_{ISLAND_NAME.lower()}.png"), dpi=300)
plt.close()


# -----------------------------
# 8. Extrema and midpoints (last 10 years)
# -----------------------------
data_last_10_years = data[data.index >= start_date]
series_last_10 = data_last_10_years["Average"]

extrema = find_extrema(series_last_10, n_abs=2)

max_indices = [extrema["real_max"]] + extrema["top_max_abs"]
min_indices = [extrema["real_min"]] + extrema["top_min_abs"]

extrema_indices = sorted(list(set([i for i in max_indices + min_indices if i is not None])))

mid_indices = find_midpoints(series_last_10, start_date)

all_explanation_indices = sorted(set(extrema_indices + mid_indices))


# -----------------------------
# 9. Interpretability (SHAP + LIME)
# -----------------------------
generate_interpretability_outputs(
    X=X,
    rf_model=rf_model,
    xgb_model=xgb_model,
    indices=all_explanation_indices,
    output_dir=EXPLAIN_DIR
)


# -----------------------------
# 10. Plot time series with extrema
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(data_last_10_years.index, series_last_10, label=SPEI_SCALE)

if max_indices:
    plt.scatter(max_indices, series_last_10.loc[max_indices], color="red", label="Maxima", zorder=5, s=50)
if min_indices:
    plt.scatter(min_indices, series_last_10.loc[min_indices], color="green", label="Minima", zorder=5, s=50)
if mid_indices:
    plt.scatter(mid_indices, series_last_10.loc[mid_indices], color="blue", label="Midpoints", zorder=5, s=50)

plt.title(f"{SPEI_SCALE} Time Series with Highlighted Extrema and Midpoints (Last 10 Years) in {ISLAND_NAME}")
plt.xlabel("Date")
plt.ylabel(SPEI_SCALE)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f"time_series_with_extrema_last_10_years_{ISLAND_NAME.lower()}.png"), dpi=300)
plt.close()

print("Analysis complete. Check the 'output' directory for generated files and visualizations.")
