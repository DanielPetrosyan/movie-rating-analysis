"""
Step 4 - Feature Engineering
movie-rating-analysis

Reads data/cleaned_movies.csv, engineers features for modeling,
saves data/movies_features.csv (used as input for Step 5 baseline model).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_IN = Path("data/cleaned_movies.csv")
DATA_OUT = Path("data/movies_features.csv")

REFERENCE_YEAR = 2026  # matches the latest release_year in the dataset
TOP_N_GENRES = 10      # keep the N most common genres as their own columns

df = pd.read_csv(DATA_IN)

# -----------------------------------------------------------------
# 1. Log transforms
#    popularity and vote_count are right-skewed (see Step 3 EDA) -
#    log1p compresses large values so models aren't dominated by a
#    handful of extreme outliers. log1p (not log) handles 0 safely.
# -----------------------------------------------------------------
df["log_popularity"] = np.log1p(df["popularity"])
df["log_vote_count"] = np.log1p(df["vote_count"])

# -----------------------------------------------------------------
# 2. Movie age
#    Raw release_year (e.g. 1994) isn't very useful to a model on
#    its own - "age in years" is a more natural numeric signal.
# -----------------------------------------------------------------
df["movie_age"] = REFERENCE_YEAR - df["release_year"]

# -----------------------------------------------------------------
# 3. Language flag
#    ~90% of movies are English; one-hot encoding every language
#    would create many near-empty columns. A single binary flag
#    captures most of the signal.
# -----------------------------------------------------------------
df["is_english"] = (df["original_language"] == "en").astype(int)

# -----------------------------------------------------------------
# 4. Number of genres per movie
#    genres column looks like "Action|Adventure|Fantasy"
# -----------------------------------------------------------------
df["num_genres"] = df["genres"].apply(lambda x: len(str(x).split("|")))

# -----------------------------------------------------------------
# 5. Decade as a number (2020s -> 2020) instead of text
# -----------------------------------------------------------------
df["decade_start"] = df["decade"].str.replace("s", "", regex=False).astype(int)

# -----------------------------------------------------------------
# 6. Primary genre one-hot encoding (top N, rest grouped as "Other")
#    Keeps the feature space small and avoids ultra-rare genre
#    columns that are mostly zeros.
# -----------------------------------------------------------------
top_genres = df["primary_genre"].value_counts().nlargest(TOP_N_GENRES).index
df["primary_genre_grouped"] = df["primary_genre"].where(
    df["primary_genre"].isin(top_genres), other="Other"
)
genre_dummies = pd.get_dummies(df["primary_genre_grouped"], prefix="genre").astype(int)
df = pd.concat([df, genre_dummies], axis=1)

# -----------------------------------------------------------------
# Save
# -----------------------------------------------------------------
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_OUT, index=False)

print("Shape:", df.shape)
print("\nNew/engineered columns:")
new_cols = [
    "log_popularity", "log_vote_count", "movie_age", "is_english",
    "num_genres", "decade_start"
] + list(genre_dummies.columns)
print(new_cols)
print(f"\nSaved engineered dataset to {DATA_OUT}")
