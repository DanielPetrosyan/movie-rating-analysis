"""
Step 6 - Model Comparison
movie-rating-analysis

Reads data/movies_features.csv, trains three models on the same
train/test split used in Step 5 (Linear Regression, Random Forest,
Gradient Boosting), compares them on R^2 / MAE / RMSE, and saves
a comparison table + bar chart.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/movies_features.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

feature_cols = (
    ["log_popularity", "log_vote_count", "movie_age", "is_english", "num_genres", "decade_start"]
    + [c for c in df.columns if c.startswith("genre_")]
)
target_col = "vote_average"

X = df[feature_cols]
y = df[target_col]

# Same split as Step 5 (random_state=42) so results are directly comparable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------------
# Models to compare
#    - Linear Regression: the Step 5 baseline (repeated here so all
#      three numbers live in one table)
#    - Random Forest: captures non-linear relationships and feature
#      interactions that a linear model can't
#    - Gradient Boosting: builds trees sequentially, often squeezes
#      out a bit more accuracy than Random Forest on tabular data
# -----------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({"model": name, "r2": r2, "mae": mae, "rmse": rmse})

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print("=== Model Comparison (test set) ===")
print(results_df.round(4).to_string(index=False))

# -----------------------------------------------------------------
# Feature importance from the best tree-based model (helps sanity-check
# against the Linear Regression coefficients from Step 5)
# -----------------------------------------------------------------
best_tree_name = results_df[results_df["model"] != "Linear Regression"].iloc[0]["model"]
best_tree_model = models[best_tree_name]
importances = pd.Series(best_tree_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop feature importances ({best_tree_name}):")
print(importances.head(8).round(3))

# -----------------------------------------------------------------
# Bar chart: R^2 comparison across models
# -----------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52"]
plt.bar(results_df["model"], results_df["r2"], color=colors[:len(results_df)])
plt.ylabel("R^2 (test set)")
plt.title("Model Comparison: R^2 Score")
for i, v in enumerate(results_df["r2"]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_model_comparison_r2.png")
plt.close()

# -----------------------------------------------------------------
# Save comparison table
# -----------------------------------------------------------------
results_df.to_csv("reports/model_comparison.csv", index=False)
print(f"\nSaved comparison table to reports/model_comparison.csv")
print(f"Saved chart to {FIG_DIR / '08_model_comparison_r2.png'}")
