"""
Day 1 — Fetch movie data from TMDB (The Movie Database) API.

Pulls popular movies with their ratings, genres, budget, revenue, and
release info, and saves them to data/raw_movies.csv.

Setup:
    1. Create a .env file in this folder with:
           TMDB_API_KEY=your_key_here
    2. pip install -r requirements.txt

Usage:
    python fetch_movies.py --pages 40 --out data/raw_movies.csv
    (each page = 20 movies, so 40 pages = ~800 movies)
"""

import argparse
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"


def get_genre_map() -> dict:
    """Fetch the genre id -> name mapping once, so we can label movies
    with real genre names instead of just numeric ids."""
    resp = requests.get(
        f"{BASE_URL}/genre/movie/list",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=15,
    )
    resp.raise_for_status()
    genres = resp.json()["genres"]
    return {g["id"]: g["name"] for g in genres}


def fetch_page(page: int, sort_by: str) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/discover/movie",
        params={
            "api_key": API_KEY,
            "sort_by": sort_by,
            "page": page,
            "include_adult": "false",
            "vote_count.gte": 50,  # skip movies with almost no votes (noisy ratings)
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=40, help="each page = ~20 movies")
    parser.add_argument("--out", default="data/raw_movies.csv")
    parser.add_argument("--sort-by", default="popularity.desc",
                         help="e.g. popularity.desc, vote_average.desc, revenue.desc")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: TMDB_API_KEY not found. Create a .env file with:")
        print("  TMDB_API_KEY=your_key_here")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Fetching genre list...")
    genre_map = get_genre_map()
    print(f"Got {len(genre_map)} genres")

    all_movies = []
    seen_ids = set()

    for page in range(1, args.pages + 1):
        print(f"Fetching page {page}/{args.pages}...")
        try:
            results = fetch_page(page, args.sort_by)
        except requests.RequestException as e:
            print(f"  Failed on page {page}: {e}")
            continue

        if not results:
            print("  No more results, stopping early.")
            break

        for m in results:
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])
            genre_names = [genre_map.get(gid, "Unknown") for gid in m.get("genre_ids", [])]
            all_movies.append({
                "id": m["id"],
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "genres": "|".join(genre_names),
                "original_language": m.get("original_language"),
                "popularity": m.get("popularity"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "adult": m.get("adult"),
            })

        time.sleep(args.delay)

    df = pd.DataFrame(all_movies)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} movies to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
