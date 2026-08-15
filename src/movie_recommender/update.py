#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:06:13 2026

@author: dina.deifallah
"""

# src/movie_recommender/update.py
# Renewing the model from new user input.
#
# The task requires being able to RENEW recommendations from new input.
# There are two kinds of renewal, and this project does both:
#   1. Real-time: a new user picks movies and gets recommendations on the spot
#      (handled by model.recommend in the Streamlit app -- no retraining).
#   2. Learning from new data: new ratings are saved and the model is
#      retrained so future recommendations reflect them. That is this file --
#      the step you would schedule to run regularly.

from pathlib import Path

import pandas as pd

from src.movie_recommender import get_logger
from src.movie_recommender.config import load_config
from src.movie_recommender.pipeline import run_pipeline

logger = get_logger(__name__)


def append_new_ratings(config, new_ratings):
    """Save newly collected ratings to the new-ratings file."""
    # Check the new ratings have the columns we expect.
    expected = {"user_id", "movie_id", "rating"}
    if not expected.issubset(new_ratings.columns):
        raise ValueError(f"new_ratings must have columns {expected}")

    path = Path(config["paths"]["new_ratings"])
    path.parent.mkdir(parents=True, exist_ok=True)

    new_ratings = new_ratings[["user_id", "movie_id", "rating"]]
    # Add to the file if it exists, otherwise create it.
    if path.exists():
        combined = pd.concat([pd.read_csv(path), new_ratings], ignore_index=True)
    else:
        combined = new_ratings

    combined.to_csv(path, index=False)
    logger.info("Saved %d new ratings (file now has %d).",
                len(new_ratings), len(combined))


def renew_model(new_ratings, config=None):
    """Add new ratings and retrain the model on the updated data."""
    if config is None:
        config = load_config()
    logger.info("=== Renewing model from %d new ratings ===", len(new_ratings))
    append_new_ratings(config, new_ratings)
    return run_pipeline(config)