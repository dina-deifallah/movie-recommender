#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:30:49 2026

@author: dina.deifallah
"""

# src/movie_recommender/preprocess.py
# Cleans the ratings and builds the user x movie matrix the model needs.

from src.movie_recommender import get_logger

logger = get_logger(__name__)


def clean_ratings(ratings, config):
    """Remove duplicates and users/movies with too few ratings."""
    pre = config["preprocess"]
    before = len(ratings)

    # If a user rated the same movie twice, keep only their last rating.
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    # Keep only movies that enough people have rated.
    movie_counts = ratings["movie_id"].value_counts()
    popular_movies = movie_counts[movie_counts >= pre["min_ratings_per_movie"]].index
    ratings = ratings[ratings["movie_id"].isin(popular_movies)]

    # Keep only users who have rated enough movies.
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= pre["min_ratings_per_user"]].index
    ratings = ratings[ratings["user_id"].isin(active_users)]

    logger.info("Cleaned ratings: %d -> %d rows.", before, len(ratings))
    return ratings


def build_user_movie_matrix(ratings):
    """Turn the long ratings table into a user x movie grid."""
    matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0,
    )
    logger.info("Built matrix: %d users x %d movies.",
                matrix.shape[0], matrix.shape[1])
    return matrix