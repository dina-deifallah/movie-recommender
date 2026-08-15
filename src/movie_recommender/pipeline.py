#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:48:20 2026

@author: dina.deifallah
"""

# src/movie_recommender/pipeline.py
# Runs the whole thing end to end: load -> clean -> evaluate -> train -> save.

from src.movie_recommender import get_logger
from src.movie_recommender.config import load_config
from src.movie_recommender.data import load_raw_data
from src.movie_recommender.preprocess import clean_ratings, build_user_movie_matrix
from src.movie_recommender.evaluate import evaluate_model, log_metrics
from src.movie_recommender.model import ItemItemRecommender

logger = get_logger(__name__)


def run_pipeline(config=None):
    """Run the full training pipeline and return the trained model."""
    if config is None:
        config = load_config()

    logger.info("=== Pipeline started ===")

    # 1. Get the data.
    movies, ratings = load_raw_data(config)

    # 2. Clean it.
    ratings = clean_ratings(ratings, config)

    # 3. Check quality on a held-out split, and record the numbers.
    metrics = evaluate_model(ratings, config)
    log_metrics(config, metrics)

    # 4. Train the final model on ALL the cleaned data.
    matrix = build_user_movie_matrix(ratings)
    model = ItemItemRecommender(
        min_common_users=config["model"]["min_common_users"],
        top_k=config["model"]["top_k"],
    ).fit(matrix)

    # 5. Save the model so the app and CLI can use it later.
    model.save(config["paths"]["model_file"])

    logger.info("=== Pipeline finished ===")
    return model