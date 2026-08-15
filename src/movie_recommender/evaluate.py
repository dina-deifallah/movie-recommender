#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:48:03 2026

@author: dina.deifallah
"""

# src/movie_recommender/evaluate.py
# Measures how good the recommendations are, and records the numbers.

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.movie_recommender import get_logger
from src.movie_recommender.model import ItemItemRecommender
from src.movie_recommender.preprocess import build_user_movie_matrix

logger = get_logger(__name__)


def split_ratings(ratings, test_fraction, seed=42):
    """Randomly split ratings into a training part and a test part."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(ratings)) < test_fraction
    train = ratings[~is_test]
    test = ratings[is_test]
    return train, test


def evaluate_model(ratings, config):
    """Train on part of the data and check recommendations on the rest."""
    ev = config["evaluation"]
    k = ev["k"]
    like = ev["like_threshold"]

    # Train a model on the training part only.
    train, test = split_ratings(ratings, ev["test_fraction"])
    matrix = build_user_movie_matrix(train)
    model = ItemItemRecommender(
        min_common_users=config["model"]["min_common_users"], top_k=k
    ).fit(matrix)

    # For each user, collect the movies they liked in train and in test.
    liked_train = train[train["rating"] >= like].groupby("user_id")["movie_id"].apply(list)
    liked_test = test[test["rating"] >= like].groupby("user_id")["movie_id"].apply(set)

    precisions = []
    recalls = []
    for user_id, relevant in liked_test.items():
        liked = liked_train.get(user_id, [])
        if not liked or not relevant:
            continue
        recommended = set(model.recommend(liked, top_k=k))
        if not recommended:
            continue
        hits = len(recommended & relevant)   # recommendations they actually liked
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    metrics = {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "k": k,
        "users_evaluated": len(precisions),
    }
    logger.info("precision@%d = %.3f | recall@%d = %.3f (%d users)",
                k, metrics["precision_at_k"], k, metrics["recall_at_k"],
                metrics["users_evaluated"])
    return metrics


def log_metrics(config, metrics):
    """Append the metrics to a JSON file, with a timestamp."""
    path = Path(config["paths"]["metrics_file"])
    path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if path.exists():
        history = json.loads(path.read_text())

    record = {"timestamp": datetime.now(timezone.utc).isoformat()}
    record.update(metrics)
    history.append(record)
    path.write_text(json.dumps(history, indent=2))
    logger.info("Logged metrics to %s (%d runs recorded).", path, len(history))