"""
Step 5 - Baseline Model
movie-rating-analysis

Reads data/movies_features.csv, trains a Linear Regression baseline
to predict vote_average, evaluates on a held-out test set, and saves
metrics to reports/metrics.json (Step 6 will add more models to compare
against this baseline).
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/movies_features.csv")
METRICS_PATH = Path("reports/metrics.json")
RANDOM_STATE = 42  # fixed seed -> reproducible train/test split every run

df = pd.read_csv(DATA_PATH)

# -----------------------------------------------------------------
# Feature selection
# NOTE: log_vote_count is included as a feature. This is a
# deliberate choice, not an oversight: vote_count is only available
# *after* a movie has accumulated ratings, so this baseline explains
# rating patterns in the existing dataset rather than predicting the
# rating of a brand-new, unreleased movie. Documented here and in
# the README so the assumption is explicit.
# -----------------------------------------------------------------
genre_cols = [c for c in df.columns if c.startswith("genre_")]
feature_cols = [
    "log_popularity", "log_vote_count", "movie_age",
    "is_english", "num_genres", "decade_start",
] + genre_cols

TARGET = "vote_average"

X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"R^2:  {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# -----------------------------------------------------------------
# Coefficients - for a linear model these double as a rough
# feature-importance view (features are on different scales, so
# this is directional, not a precise ranking).
# -----------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_
}).sort_values("coefficient", ascending=False)

print("\nCoefficients (sorted):")
print(coef_df.to_string(index=False))

# -----------------------------------------------------------------
# Save metrics for Step 6 comparison
# -----------------------------------------------------------------
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
metrics = {}
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())

metrics["linear_regression_baseline"] = {
    "r2": round(r2, 4),
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "features": feature_cols,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"\nSaved metrics to {METRICS_PATH}")
