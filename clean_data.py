"""
Day 2 — Data cleaning for the movie rating analysis project.

Takes data/raw_movies.csv and produces data/cleaned_movies.csv, ready
for EDA and modeling.

What this does:
1. Parses release_date into release_year and decade (more useful for
   modeling than a raw date string).
2. Extracts a primary_genre (the first listed genre) as a simple
   single-label feature, while keeping the full pipe-separated genre
   list too (useful for one-hot encoding later).
3. Drops the 'adult' column — every movie in our data is non-adult
   (we filtered for that when fetching), so it has zero variance and
   is useless as a feature.
4. Removes duplicate movie ids, if any.
5. Sanity-checks vote_average/vote_count/popularity ranges.

Usage:
    python clean_data.py --in data/raw_movies.csv --out data/cleaned_movies.csv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="data/raw_movies.csv")
    parser.add_argument("--out", dest="outfile", default="data/cleaned_movies.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    n_start = len(df)
    print(f"Loaded {n_start} raw movies")

    # 1. Drop duplicate movie ids (shouldn't be any, but good practice)
    df = df.drop_duplicates(subset="id")
    print(f"After removing duplicate ids: {len(df)}")

    # 2. Parse release_date -> release_year, decade
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["release_date"])  # drop anything with an unparseable date
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with unparseable release dates")

    df["release_year"] = df["release_date"].dt.year
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    # 3. Primary genre (first in the pipe-separated list) — a simple
    #    single-label feature, useful alongside the full genre list.
    df["primary_genre"] = df["genres"].str.split("|").str[0]

    # 4. Drop the 'adult' column — zero variance, not useful
    if df["adult"].nunique() <= 1:
        print(f"Dropping 'adult' column (all values are {df['adult'].iloc[0]}, no variance)")
        df = df.drop(columns=["adult"])

    # 5. Sanity check numeric ranges (nothing to remove here, just reporting —
    #    the data already came pre-filtered from the API with vote_count >= 50)
    print(f"\nvote_average range: {df['vote_average'].min()} - {df['vote_average'].max()}")
    print(f"vote_count range: {df['vote_count'].min()} - {df['vote_count'].max()}")
    print(f"release_year range: {df['release_year'].min()} - {df['release_year'].max()}")

    # 6. Final column order
    df = df[
        [
            "id", "title", "release_date", "release_year", "decade",
            "primary_genre", "genres", "original_language",
            "popularity", "vote_average", "vote_count",
        ]
    ]

    df.to_csv(args.outfile, index=False)
    print(f"\nSaved {len(df)} cleaned movies to {args.outfile}")
    print(f"({n_start - len(df)} rows removed total)")

    print("\nTop 10 primary genres:")
    print(df["primary_genre"].value_counts().head(10))
    print("\nMovies per decade:")
    print(df["decade"].value_counts().sort_index())


if __name__ == "__main__":
    main()
