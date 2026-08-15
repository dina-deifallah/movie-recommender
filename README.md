#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:21:53 2026

@author: dina.deifallah
"""

# Movie Recommender — an example ML *system*

A small but complete movie recommendation **system**, built as a worked
example for the module *Project: Machine Learning Systems Design
(DLMDSPMLSD01), Task 1*. It is built the way the rubric rewards: a repeatable
pipeline, a model you can save/load/renew, evaluation, and a web interface —
not a single notebook.

> The machine-learning part (an item-item similarity matrix) is about 40 lines.
> Everything else — config, ingestion, a pipeline, an update path, evaluation,
> a UI, tests — is the *system*. In this module the system is the deliverable.

## What it does

Pick a few movies you like; it recommends similar ones, using item-item
collaborative filtering (two movies are "similar" when the same people rate
them alike).

It meets the task's **mandatory** requirements:
- **Renews recommendations from new input** — in real time (the app) and by
  retraining on newly collected ratings (`update.py`).
- **Lives in a Git repository** with this README and a commit history.

## Where the data comes from

The [MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/)
dataset (~100k ratings, 610 users, 9.7k movies), downloaded automatically from
GroupLens on first run — no account or API key needed. Data files are **not**
committed to git.

## The components

| Component     | File                                     |
|---------------|------------------------------------------|
| Configuration | `config.py` + `config.yaml`              |
| Ingestion     | `data.py` (download + load)              |
| Preprocessing | `preprocess.py` (clean + build matrix)   |
| Model         | `model.py` (fit / recommend / save / load) |
| Evaluation    | `evaluate.py` (precision@k, recall@k)    |
| Orchestration | `pipeline.py` (one full run)             |
| Renewal       | `update.py`, `scripts/scheduled_update.py` |
| Interface     | `app/streamlit_app.py`                   |

## Quickstart

```bash
pip install -r requirements.txt

# Train the model (downloads the data on first run)
python -c "from src.movie_recommender.pipeline import run_pipeline; run_pipeline()"

# Launch the web app
streamlit run app/streamlit_app.py
```

## Running the tests

```bash
pytest -q
```

## Reviewer access (for the assignment)

Keep the repository **private** during Phases 1–2 and grant the reviewer
access; make it public only at finalisation.
- GitHub: *Settings → Collaborators → Add people*
- GitLab: *Manage → Members → Invite member*

Put the repository URL in your Phase 2 summary text.

---

*Teaching example. The model is kept deliberately simple so the systems-design
scaffolding around it is easy to read.*