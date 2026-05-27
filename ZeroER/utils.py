"""Extras that live *outside* the pyJedAI package — dataset I/O and metrics
for the ZeroER++ experiments.

Framework-light by design: only pandas plus a pyJedAI ``Data`` builder, no
blocking/matching logic. Keeping dataset quirks (attribute selection, column
renames, ground-truth leaks) here means the model and wrapper layers never
have to know which dataset they are looking at.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from pyjedai.datamodel import Data

# Self-contained: the five gold-standard datasets live next to this file under
# zeroer-experiments/datasets/, resolved relative to this file (cwd-independent).
BENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "datasets"))

# Fraction of each benchmark kept for the Optuna *tuning* partition (see
# ``load_data(partition=True)``). The study tunes on this slice only; the best
# params are then retrained on the full dataset. Per-dataset overridable below.
DEFAULT_TUNE_FRACTION = 0.3

# Per-dataset metadata: matchable attributes, the tuning-partition fraction, plus
# any column rename needed to align the two tables on a shared name:
#   - fodors_zagats: 'type'/'class' dropped — 'class' is a ground-truth leak.
#   - amazon_googleproducts: Amazon's 'title' and Google's 'name' are the same
#     field under different names, so the right table is renamed 'name'->'title'.
DATASETS = {
    "fodors_zagats":         dict(attributes=["name", "addr", "city", "phone"],
                                  tune_fraction=0.3),
    "dblp_acm":              dict(attributes=["title", "authors", "venue", "year"],
                                  tune_fraction=0.3),
    "dblp_scholar":          dict(attributes=["title", "authors", "venue", "year"],
                                  tune_fraction=0.3),
    "abt_buy":               dict(attributes=["name", "description", "price"],
                                  tune_fraction=0.3),
    "amazon_googleproducts": dict(attributes=["title", "description", "manufacturer", "price"],
                                  tune_fraction=0.3, rename_2={"name": "title"}),
}


def _subsample(
    left: pd.DataFrame,
    right: pd.DataFrame,
    gt: pd.DataFrame,
    fraction: float,
    seed: int,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair-preserving subsample of a clean-clean ER benchmark.

    Parameters
    ----------
    left, right
        Entity tables for the two sides of the clean-clean ER task. Each table
        is expected to contain a unique identifier column named ``id_col``.

    gt
        Ground-truth matches dataframe. The first two columns are assumed to be
        the left/right entity ids respectively (e.g. ``ltable_id``,
        ``rtable_id``).

    fraction
        Fraction of matched pairs and singleton entities to retain. Must lie in
        ``(0, 1]``.

    seed
        Random seed used for deterministic sampling.

    id_col
        Name of the entity-id column in both entity tables.

    Returns
    -------
    (left_s, right_s, gt_s)
        Subsampled left/right tables together with a recomputed ground-truth
        dataframe containing only surviving match pairs.

    Notes
    -----
    Idea here is to preserve the matched:unmatched ratio approximately (well, linearly)
    with the requested fraction, making tuning results transfer more generalisable
    to the full dataset.

    Steps:
      (1) sample ~``fraction`` of the gold pairs and keep both endpoints;
      (2) sample ~``fraction`` of unmatched singleton entities on each side;
      (3) recompute the ground truth from surviving endpoints.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")

    lcol, rcol = gt.columns[:2]

    # (1) ~fraction of matched pairs -> keep both endpoints
    gt_sampled = gt.sample(frac=fraction, random_state=seed)

    keep_l = (
        set(gt_sampled[lcol])
        | set(
            left.loc[~left[id_col].isin(gt[lcol]), id_col]
                .sample(frac=fraction, random_state=seed)
        )
    )

    keep_r = (
        set(gt_sampled[rcol])
        | set(
            right.loc[~right[id_col].isin(gt[rcol]), id_col]
                 .sample(frac=fraction, random_state=seed)
        )
    )

    # (2) assemble partition tables from surviving endpoints
    left_s = (
        left.loc[lambda df: df[id_col].isin(keep_l)]
            .reset_index(drop=True)
    )

    right_s = (
        right.loc[lambda df: df[id_col].isin(keep_r)]
             .reset_index(drop=True)
    )

    # (3) recompute GT consistent with surviving endpoints
    gt_s = (
        gt.loc[lambda df: df[lcol].isin(keep_l) & df[rcol].isin(keep_r)]
          .reset_index(drop=True)
    )

    return left_s, right_s, gt_s

def load_data(
    name: str,
    base: str = BENCH_DIR,
    partition: bool = False,
    seed: int = 0
) -> Data:
    """Load a zeroer-bench dataset into a pyJedAI clean-clean ``Data`` object.

    ``partition=True`` returns only the seeded ~``tune_fraction`` slice (see
    ``_subsample``) for hyperparameter tuning; the default returns the full
    dataset, which is the path for retraining the best params. The partition
    gets a distinct ``dataset_name`` so its PLM embeddings cache separately from
    the full-data embeddings (otherwise the full ``.npy`` would be indexed with
    the partition's smaller row indices, silently returning wrong vectors).
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    d = os.path.join(base, name)
    left = pd.read_csv(os.path.join(d, "left.csv")).astype(str)
    right = pd.read_csv(os.path.join(d, "right.csv")).astype(str)
    if spec.get("rename_1"):
        left = left.rename(columns=spec["rename_1"])
    if spec.get("rename_2"):
        right = right.rename(columns=spec["rename_2"])
    gt = pd.read_csv(os.path.join(d, "matches.csv")).astype(str)
    attrs = spec["attributes"]

    name_l, name_r = f"{name}:left", f"{name}:right"
    if partition:
        frac = spec.get("tune_fraction", DEFAULT_TUNE_FRACTION)
        left, right, gt = _subsample(left, right, gt, frac, seed)
        tag = f"p{int(round(frac * 100))}s{seed}"
        name_l, name_r = f"{name}:left:{tag}", f"{name}:right:{tag}"

    return Data(
        dataset_1=left, id_column_name_1="id", attributes_1=attrs,
        dataset_name_1=name_l,
        dataset_2=right, id_column_name_2="id", attributes_2=attrs,
        dataset_name_2=name_r,
        ground_truth=gt,
    )


def _count_gold_pairs_present(adj, data) -> int:
    """Count gold-standard pairs present in an adjacency-like container.

    Works for both a networkx ``Graph`` (the match graph) and a ``{id: set}``
    blocks dict — both answer ``i1 in adj`` and ``i2 in adj[i1]`` — so the two
    metrics below share exactly one membership test.
    """
    n = 0
    for _, (id1, id2) in data.ground_truth.iterrows():
        i1 = data._ids_mapping_1[id1]
        i2 = data._ids_mapping_2[id2]
        if (i1 in adj and i2 in adj[i1]) or (i2 in adj and i1 in adj[i2]):
            n += 1
    return n


def precision_recall_f1(graph, data) -> Tuple[float, float, float]:
    """End-to-end P/R/F1 of a match graph against the full gold standard.

    Recall denominator is the *complete* ground truth, so matches dropped by
    blocking count against recall (the statistically sound estimand when the
    blocker is itself a variable being compared).
    """
    tp = _count_gold_pairs_present(graph, data)
    pred = graph.number_of_edges()
    truth = len(data.ground_truth)
    p = tp / pred if pred else 0.0
    r = tp / truth if truth else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def blocking_recall(blocks, data) -> float:
    """Pair completeness: the fraction of gold matches that survive blocking.

    Computed on the candidate-pair structure *before* the matcher runs, so it
    isolates the matches the blocker discards (unrecoverable) from those the
    matcher later rejects. It is therefore the ceiling that end-to-end recall
    can reach for a given blocking configuration — the "don't drop true pairs to
    begin with" objective of the sweep.
    """
    if not blocks:
        return 0.0
    truth = len(data.ground_truth)
    return _count_gold_pairs_present(blocks, data) / truth if truth else 0.0
