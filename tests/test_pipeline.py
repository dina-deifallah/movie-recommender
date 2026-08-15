#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:17:35 2026

@author: dina.deifallah
"""

# tests/test_pipeline.py
# Small tests for the core logic. Run with:  pytest -q
#
# These use tiny hand-built data (no download, no network), so they run in a
# blink and check the things that actually break a pipeline: filtering,
# matrix shape, a model that never recommends its own inputs, and save/load.

import pandas as pd

from src.movie_recommender.model import ItemItemRecommender
from src.movie_recommender.preprocess import clean_ratings, build_user_movie_matrix


def small_ratings():
    """A tiny dataset where movies 10 and 11 are clearly liked together."""
    rows = [
        (1, 10, 5.0), (1, 11, 5.0), (1, 12, 1.0),
        (2, 10, 4.0), (2, 11, 5.0), (2, 12, 2.0),
        (3, 10, 5.0), (3, 11, 4.0), (3, 13, 5.0),
        (4, 12, 5.0), (4, 13, 4.0), (4, 11, 1.0),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


# A minimal config so we don't need config.yaml in the tests.
SMALL_CONFIG = {"preprocess": {"min_ratings_per_movie": 1, "min_ratings_per_user": 1}}


def test_clean_ratings_drops_duplicates():
    doubled = pd.concat([small_ratings(), small_ratings()], ignore_index=True)
    cleaned = clean_ratings(doubled, SMALL_CONFIG)
    assert not cleaned.duplicated(subset=["user_id", "movie_id"]).any()


def test_build_matrix_shape():
    matrix = build_user_movie_matrix(small_ratings())
    assert matrix.shape[0] == 4      # 4 users
    assert matrix.shape[1] == 4      # movies 10, 11, 12, 13


def test_model_excludes_inputs():
    matrix = build_user_movie_matrix(small_ratings())
    model = ItemItemRecommender(min_common_users=1, top_k=3).fit(matrix)
    recs = model.recommend([10], top_k=3)
    assert 10 not in recs                          # never recommends its input
    assert all(m in matrix.columns for m in recs)  # only recommends known movies


def test_save_and_load_roundtrip(tmp_path):
    matrix = build_user_movie_matrix(small_ratings())
    model = ItemItemRecommender(min_common_users=1).fit(matrix)
    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = ItemItemRecommender.load(path)
    assert reloaded.recommend([10]) == model.recommend([10])