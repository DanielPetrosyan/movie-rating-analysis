"""
Step 3 - Exploratory Data Analysis
movie-rating-analysis

Reads data/cleaned_movies.csv, produces summary stats and saves
6 figures to reports/figures/, used to inform feature engineering
in Step 4.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/cleaned_movies.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------
# 0. Quick console summary
# ---------------------------------------------------------------
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nNumeric summary:\n", df[["popularity", "vote_average", "vote_count"]].describe())
print("\nCorrelation (numeric features):")
print(df[["popularity", "vote_average", "vote_count", "release_year"]].corr(numeric_only=True).round(2))

# ---------------------------------------------------------------
# 1. Distribution of vote_average
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.histplot(df["vote_average"], bins=25, kde=True, color="#4C72B0")
plt.axvline(df["vote_average"].mean(), color="red", linestyle="--", label=f"mean = {df['vote_average'].mean():.2f}")
plt.title("Distribution of Vote Average")
plt.xlabel("Vote Average")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_vote_average_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 2. Average rating by primary_genre (genres with n >= 10 only)
# ---------------------------------------------------------------
genre_counts = df["primary_genre"].value_counts()
valid_genres = genre_counts[genre_counts >= 10].index
genre_df = df[df["primary_genre"].isin(valid_genres)]
genre_order = genre_df.groupby("primary_genre")["vote_average"].mean().sort_values(ascending=False).index

plt.figure(figsize=(8, 6))
sns.boxplot(data=genre_df, y="primary_genre", x="vote_average", order=genre_order,
            hue="primary_genre", palette="viridis", legend=False)
plt.title("Vote Average by Primary Genre (n >= 10 movies)")
plt.xlabel("Vote Average")
plt.ylabel("Primary Genre")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_rating_by_genre.png")
plt.close()

# ---------------------------------------------------------------
# 3. Average rating by decade
# ---------------------------------------------------------------
decade_order = sorted(df["decade"].unique())
decade_stats = df.groupby("decade")["vote_average"].agg(["mean", "count"]).reindex(decade_order)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(decade_stats.index, decade_stats["mean"], color="#55A868")
ax1.set_ylabel("Mean Vote Average")
ax1.set_xlabel("Decade")
ax1.set_title("Average Rating by Decade (bars) with Movie Count (line)")
plt.xticks(rotation=45)

ax2 = ax1.twinx()
ax2.plot(decade_stats.index, decade_stats["count"], color="black", marker="o")
ax2.set_ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_rating_by_decade.png")
plt.close()

# ---------------------------------------------------------------
# 4. Popularity vs Rating
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="popularity", y="vote_average", alpha=0.6, edgecolor=None)
plt.xscale("log")
plt.title("Popularity vs Vote Average (log scale)")
plt.xlabel("Popularity (log scale)")
plt.ylabel("Vote Average")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_popularity_vs_rating.png")
plt.close()

# ---------------------------------------------------------------
# 5. Vote count vs Rating (checks whether low-vote movies are noisy)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="vote_count", y="vote_average", alpha=0.6, edgecolor=None, color="#C44E52")
plt.xscale("log")
plt.title("Vote Count vs Vote Average (log scale)")
plt.xlabel("Vote Count (log scale)")
plt.ylabel("Vote Average")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_votecount_vs_rating.png")
plt.close()

# ---------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------
plt.figure(figsize=(5.5, 4.5))
corr = df[["popularity", "vote_average", "vote_count", "release_year"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_correlation_heatmap.png")
plt.close()

print(f"\nSaved 6 figures to {FIG_DIR}/")
