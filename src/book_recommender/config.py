#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:35:04 2026

@author: dina.deifallah
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_ratings: Path
    raw_books: Path
    processed_matrix: Path
    model_file: Path
    metrics_file: Path


@dataclass(frozen=True)
class Config:
    paths: Paths
    simulation: dict
    preprocess: dict
    model: dict
    evaluation: dict


def _resolve(path_str: str) -> Path:
    return (PROJECT_ROOT / path_str).resolve()


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    paths = Paths(
        raw_ratings=_resolve(raw["paths"]["raw_ratings"]),
        raw_books=_resolve(raw["paths"]["raw_books"]),
        processed_matrix=_resolve(raw["paths"]["processed_matrix"]),
        model_file=_resolve(raw["paths"]["model_file"]),
        metrics_file=_resolve(raw["paths"]["metrics_file"]),
    )
    return Config(
        paths=paths,
        simulation=raw["simulation"],
        preprocess=raw["preprocess"],
        model=raw["model"],
        evaluation=raw["evaluation"],
    )