#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:55:47 2026

@author: dina.deifallah
"""

# app/streamlit_app.py
# Web interface for the movie recommender.
# Run from a terminal with:  streamlit run app/streamlit_app.py

import sys
from pathlib import Path

import streamlit as st

# Let this file find the 'src' package when Streamlit runs it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.movie_recommender.config import load_config
from src.movie_recommender.data import load_raw_data
from src.movie_recommender.model import ItemItemRecommender


@st.cache_resource
def load_everything():
    """Load config, movies, and the trained model once (then reuse)."""
    config = load_config()
    movies, _ = load_raw_data(config)
    model = ItemItemRecommender.load(config["paths"]["model_file"])
    return config, movies, model


# --- page setup ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("Pick a few movies you like and get recommendations.")

# --- load data + model ---
try:
    config, movies, model = load_everything()
except FileNotFoundError:
    st.error("No trained model found. Run the pipeline first "
             "(python -c \"from src.movie_recommender.pipeline import run_pipeline; run_pipeline()\").")
    st.stop()

# --- the user picks movies from a dropdown ---
title_to_id = dict(zip(movies["title"], movies["movie_id"]))
chosen_titles = st.multiselect(
    "Movies you like",
    options=sorted(title_to_id.keys()),
    max_selections=5,
)
top_k = st.slider("How many recommendations", 1, 10, 5)

# --- the button that produces recommendations ---
if st.button("Recommend", type="primary"):
    liked_ids = [title_to_id[t] for t in chosen_titles]
    rec_ids = model.recommend(liked_ids, top_k=top_k)

    if not rec_ids:
        st.warning("No recommendations found. Try selecting more movies.")
    else:
        lookup = movies.set_index("movie_id")
        st.subheader("Recommended for you")
        for movie_id in rec_ids:
            row = lookup.loc[movie_id]
            st.write(f"**{row['title']}** — {row['genres']}")