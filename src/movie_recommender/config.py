#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:35:04 2026

@author: dina.deifallah
"""

# src/book_recommender/config.py
# Reads config.yaml and returns it as a plain dictionary.

import yaml
from pathlib import Path

# The project root is two folders up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    """Load config.yaml and return it as a dictionary."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config