# Movie Rating Analysis & Prediction

Analyzing and predicting movie ratings using real data from
[TMDB (The Movie Database)](https://www.themoviedb.org/).

## Status
🚧 Day 1 — data collection in progress.

## Project plan
- [x] Day 1: Fetch movie data from TMDB API → raw CSV
- [ ] Day 2: Data cleaning
- [ ] Day 3: Exploratory data analysis
- [ ] Day 4: Feature engineering
- [ ] Day 5: Baseline model (Linear Regression)
- [ ] Day 6: Model comparison (Random Forest / Gradient Boosting)
- [ ] Day 7: Final write-up

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows; `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

You'll also need a free TMDB API key:
1. Sign up at https://www.themoviedb.org/signup
2. Go to Settings → API → request a Developer key
3. Create a `.env` file in this folder (never commit this — it's already
   in `.gitignore`) with:
   ```
   TMDB_API_KEY=your_key_here
   ```

## Usage

Fetch movie data (40 pages ≈ 800 movies is a good first run):

```bash
python fetch_movies.py --pages 40 --out data/raw_movies.csv
```

This produces `data/raw_movies.csv` with columns:

| column | meaning |
|---|---|
| id | TMDB movie id |
| title | movie title |
| release_date | release date |
| genres | pipe-separated genre list, e.g. `Action|Adventure` |
| original_language | ISO language code |
| popularity | TMDB's popularity score |
| vote_average | average user rating (0-10) — this is what we'll predict |
| vote_count | number of votes |
| adult | whether flagged as adult content |
