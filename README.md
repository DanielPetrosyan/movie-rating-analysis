# Movie Rating Analysis

An end-to-end data science project analyzing what factors (if any) correlate with
a movie's TMDB rating — from raw API data to a compared set of regression models.

Built as part of a data science / ML portfolio. Data pulled from [The Movie Database
(TMDB) API](https://www.themoviedb.org/documentation/api).

## Key Result

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Gradient Boosting** | **0.334** | 0.455 | 0.629 |
| Random Forest | 0.277 | 0.478 | 0.655 |
| Linear Regression (baseline) | 0.177 | 0.515 | 0.699 |

Gradient Boosting improved on the linear baseline by ~88% relative R², but the
overall R² staying below 0.4 is itself a finding, not a failure — see
[Limitations](#limitations--honest-takeaways) below.

## Project Structure

```
movie-rating-analysis/
├── data/
│   ├── raw_movies.csv          # raw TMDB API pull
│   ├── cleaned_movies.csv      # after cleaning
│   └── movies_features.csv     # after feature engineering
├── reports/
│   ├── figures/                # all EDA + model plots (01-08)
│   ├── baseline_metrics.txt
│   └── model_comparison.csv
├── fetch_movies.py             # Step 1 - pull data from TMDB API
├── clean_data.py                # Step 2 - cleaning
├── eda.py                       # Step 3 - exploratory data analysis
├── feature_engineering.py       # Step 4 - feature engineering
├── baseline_model.py            # Step 5 - baseline linear regression
├── model_comparison.py          # Step 6 - Random Forest / Gradient Boosting
├── requirements.txt
└── README.md
```

## Steps

**Step 1 — Data Collection** (`fetch_movies.py`)
Pulls movie metadata from the TMDB API (title, release date, genres, language,
popularity, vote average, vote count).

**Step 2 — Data Cleaning** (`clean_data.py`)
Removes duplicates, handles missing values, derives `release_year`/`decade` from
`release_date`, and produces `data/cleaned_movies.csv` — 707 movies, 0 missing
values, 0 duplicates.

**Step 3 — Exploratory Data Analysis** (`eda.py`)
Six figures covering rating distribution, rating by genre/decade, and the
relationship between popularity, vote count, and rating.

Key findings:
- Ratings are roughly normally distributed (mean ≈ 7.27), with a small cluster of
  low-vote outliers around 3.5–4.5.
- **Popularity barely correlates with rating** (r ≈ 0.07) — popular movies aren't
  necessarily well-rated ones.
- **Vote count is the strongest numeric correlate of rating** (r ≈ 0.39) — movies
  with very few votes show much more volatile ratings.
- Genre differences in mean rating exist but the distributions overlap heavily —
  genre alone is a weak predictor.
- Older decades show higher average ratings, but this is **survivorship bias**:
  TMDB only has a handful of (already well-regarded) movies from the 1930s–1980s,
  versus 300+ from the 2020s including plenty of average ones.

**Step 4 — Feature Engineering** (`feature_engineering.py`)
Log-transforms for `popularity` and `vote_count` (both right-skewed), `movie_age`,
an `is_english` flag, `num_genres`, `decade_start`, and one-hot encoded genre
columns (top 10 genres, rest grouped as "Other"). Produces `data/movies_features.csv`
(707 rows × 29 columns).

**Step 5 — Baseline Model** (`baseline_model.py`)
Linear Regression on the engineered features, 80/20 train/test split. R² = 0.177 —
confirms that the available metadata has limited predictive power on its own.

**Step 6 — Model Comparison** (`model_comparison.py`)
Random Forest and Gradient Boosting trained on the same split. Gradient Boosting
wins (R² = 0.334), driven mostly by `log_vote_count`, `movie_age`, and
`log_popularity` — the same three signals identified in Step 3's EDA, now captured
with non-linear interactions a linear model couldn't represent.

## Limitations & Honest Takeaways

R² topping out around 0.33 is expected, not a bug to chase away: a movie's rating
is driven largely by subjective quality factors — writing, acting, direction — that
simply aren't present in TMDB's metadata (popularity, genre, vote count, language).
Predicting audience *reception* from *catalog metadata* is inherently a hard,
noisy problem, and this project's value is in demonstrating that clearly through
EDA and model comparison rather than overfitting a model to squeeze out a
misleadingly high score.

If extended further, the most promising next step would be enriching the dataset
with text-based features (e.g. plot overview embeddings, cast/director history) —
which likely explain far more of the variance than release metadata does.

## How to Run

```bash
pip install -r requirements.txt

python fetch_movies.py          # Step 1 - requires a TMDB API key in .env
python clean_data.py            # Step 2
python eda.py                   # Step 3
python feature_engineering.py   # Step 4
python baseline_model.py        # Step 5
python model_comparison.py      # Step 6
```

A TMDB API key is required for Step 1 only (steps 2-6 run on the data already
included in `data/`). Create a `.env` file with:
```
TMDB_API_KEY=your_key_here
```

## Tech Stack

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, TMDB API

## Author

Daniel Petrosyan
