#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:42:12 2026

@author: dina.deifallah
"""

# src/movie_recommender/model.py
# The recommendation model: item-item collaborative filtering.
# Two movies are "similar" if the same people tend to rate them alike.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.movie_recommender import get_logger

logger = get_logger(__name__)


class ItemItemRecommender:
    """Recommends movies similar to ones a user already likes."""

    def __init__(self, min_common_users=3, top_k=5):
        # Settings for the model.
        self.min_common_users = min_common_users
        self.top_k = top_k
        # These stay empty until we call fit().
        self.similarity = None
        self.movie_ids = None

    def fit(self, matrix):
        """Learn how similar every pair of movies is."""
        # 'matrix' is users x movies. We want to compare movies to each other,
        # so we transpose to movies x users, then measure how alike each pair is.
        similarity = cosine_similarity(matrix.T.values)

        # Don't trust the similarity of two movies that share very few raters.
        rated = (matrix.values > 0).astype(int)   # 1 where a user rated a movie
        shared = rated.T @ rated                    # how many users rated BOTH movies
        similarity[shared < self.min_common_users] = 0.0

        # A movie must never recommend itself, so zero the diagonal.
        np.fill_diagonal(similarity, 0.0)

        # Store the result as a labelled table (rows/cols are movie_ids).
        self.movie_ids = matrix.columns.to_numpy()
        self.similarity = pd.DataFrame(
            similarity, index=self.movie_ids, columns=self.movie_ids
        )
        logger.info("Fitted similarity over %d movies.", len(self.movie_ids))
        return self

    def recommend(self, liked_movie_ids, top_k=None):
        """Recommend movies based on ones the user likes."""
        if self.similarity is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")

        if top_k is None:
            top_k = self.top_k

        # Keep only the liked movies the model actually knows about.
        known = [m for m in liked_movie_ids if m in self.similarity.index]
        if not known:
            logger.warning("None of the liked movies are known to the model.")
            return []

        # Add up the similarity coming from every liked movie.
        scores = self.similarity.loc[known].sum(axis=0)
        # Don't recommend movies they already told us they like.
        scores = scores.drop(index=known)
        # Keep positives, sort high-to-low, take the top ones.
        scores = scores[scores > 0].sort_values(ascending=False)
        return scores.head(top_k).index.tolist()

    def save(self, path):
        """Save the trained model to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "similarity": self.similarity,
                "movie_ids": self.movie_ids,
                "min_common_users": self.min_common_users,
                "top_k": self.top_k,
            },
            path,
        )
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path):
        """Load a model that was saved earlier."""
        data = joblib.load(Path(path))
        model = cls(min_common_users=data["min_common_users"], top_k=data["top_k"])
        model.similarity = data["similarity"]
        model.movie_ids = data["movie_ids"]
        logger.info("Loaded model with %d movies.", len(model.movie_ids))
        return model