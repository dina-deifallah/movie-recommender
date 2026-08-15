#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:22:48 2026

@author: dina.deifallah
"""

# src/book_recommender/data.py
# Downloads the MovieLens (latest-small) dataset directly from GroupLens
# (no Kaggle account needed) and loads it.
# Source: https://grouplens.org/datasets/movielens/latest/

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.movie_recommender import get_logger

logger = get_logger(__name__)


def download_dataset(config):
    """Download the MovieLens zip from GroupLens and unzip it."""
    url = config["download"]["url"]
    download_dir = Path(config["download"]["dir"])
    download_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_dir / "movielens.zip"

    logger.info("Downloading MovieLens from %s ...", url)
    urllib.request.urlretrieve(url, zip_path)   # fetch the zip

    logger.info("Unzipping ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(download_dir)              # extract into data/raw/

    logger.info("Download complete.")


def _find_file(download_dir, filename):
    """Find a file by name anywhere under the download folder.

    The zip extracts into an 'ml-latest-small/' subfolder, so we search
    recursively (rglob) instead of assuming an exact path.
    """
    matches = list(Path(download_dir).rglob(filename))
    return matches[0] if matches else None

def _add_new_ratings(config, ratings):
    """Fold in any new ratings we've collected since the original download."""
    new_path = Path(config["paths"]["new_ratings"])
    if new_path.exists():
        new = pd.read_csv(new_path)
        ratings = pd.concat([ratings, new], ignore_index=True)
        logger.info("Added %d new ratings from %s.", len(new), new_path)
    return ratings

def load_raw_data(config):
    """Load movies and ratings, downloading them first if needed."""
    download_dir = config["download"]["dir"]
    ratings_name = config["download"]["ratings_file"]
    movies_name = config["download"]["movies_file"]

    # Download only if the ratings file isn't already on disk.
    if _find_file(download_dir, ratings_name) is None:
        download_dataset(config)

    ratings_path = _find_file(download_dir, ratings_name)
    movies_path = _find_file(download_dir, movies_name)

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    
    # Print the real column names so we can confirm the mapping.
    logger.info("Raw movie columns: %s", list(movies.columns))
    logger.info("Raw rating columns: %s", list(ratings.columns))

    # --- Map the dataset's columns onto our simple standard schema ---
    movies = movies.rename(columns={
        "movieId": "movie_id",
        "title": "title",
        "genres": "genres",
    })[["movie_id", "title", "genres"]]

    ratings = ratings.rename(columns={
        "userId": "user_id",
        "movieId": "movie_id",
        "rating": "rating",
    })[["user_id", "movie_id", "rating"]]
    
    ratings = _add_new_ratings(config, ratings)

    logger.info("Loaded %d ratings and %d movies.", len(ratings), len(movies))
    return movies, ratings