#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:07:51 2026

@author: dina.deifallah
"""

# scripts/scheduled_update.py
# The script you would run on a schedule (e.g. nightly) to keep the model
# current: it collects any new ratings and retrains.
#
# Run manually:  python scripts/scheduled_update.py
# Or schedule with cron, e.g. nightly at 2am:
#   0 2 * * *  cd /path/to/movie-recommender && python scripts/scheduled_update.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script find the 'src' package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.movie_recommender import get_logger
from src.movie_recommender.update import renew_model

logger = get_logger("scheduled_update")


def collect_new_ratings():
    """Pretend to collect new ratings from users.

    In a real system, replace this with a read from wherever your app stores
    the ratings people have submitted since the last run (a database, a file,
    a queue).
    """
    rng = np.random.default_rng()
    n = 100
    return pd.DataFrame({
        "user_id": rng.integers(100000, 101000, size=n),
        "movie_id": rng.integers(1, 200, size=n),
        "rating": rng.choice([3.0, 3.5, 4.0, 4.5, 5.0], size=n),
    })


def main():
    new_ratings = collect_new_ratings()
    logger.info("Collected %d new ratings.", len(new_ratings))
    renew_model(new_ratings)
    logger.info("Renewal complete.")


if __name__ == "__main__":
    main()