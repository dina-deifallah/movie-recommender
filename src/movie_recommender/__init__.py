#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:30:27 2026

@author: dina.deifallah
"""

# src/book_recommender/__init__.py
# Makes the folder a Python package and sets up logging.

import logging

# Configure how all log messages look, once for the whole project.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name):
    """Return a logger for a module."""
    return logging.getLogger(name)