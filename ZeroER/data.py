"""Datasets and record embeddings for the ZeroER++ experiments.

Owns the benchmark registry and everything that reads from disk: the five
gold-standard datasets, their tuning partitions, and the precomputed LLM
record embeddings. Nothing here knows about blocking, matching or Optuna.
"""

from __future__ import annotations

import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd

from pyjedai.datamodel import Data

# Self-contained: the five gold-standard datasets live next to this file under
# zeroer-experiments/datasets/, resolved relative to this file (cwd-independent).
BENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "datasets"))

# Precomputed LLM record embeddings, one .npy per side per model, row-aligned
# with left.csv / right.csv (see datasets/embeddings/README.md):
#   datasets/embeddings/<dataset>/<model-stem>_{left,right}.npy
EMB_DIR = os.path.join(BENCH_DIR, "embeddings")

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
                                  tune_fraction=0.3, rename_2={"name": "title"})
}


def embedding_models(name: str, base: str = EMB_DIR) -> list:
    """Sorted model stems with precomputed embeddings for a dataset.

    A stem is the on-disk filename minus the ``_left.npy`` suffix (it starts
    with the embedding-model name, e.g. ``Qwen_Qwen3-Embedding-4B``); only
    models with both sides present count.
    """
    d = os.path.join(base, name)
    stems = sorted(os.path.basename(p)[:-len("_left.npy")]
                   for p in glob.glob(os.path.join(d, "*_left.npy")))
    return [s for s in stems
            if os.path.exists(os.path.join(d, f"{s}_right.npy"))]


def load_embeddings(name: str, model: str, data,
                    base: str = EMB_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Record vectors for ``data``'s two tables, as float32 (faiss-ready).

    The .npy files are row-aligned with the *full* left.csv / right.csv;
    ``data`` may be a subsampled partition, so rows are re-selected by matching
    the partition's id column against the full table's ids — an identity
    reindex on the full dataset.
    """
    d = os.path.join(base, name)
    out = []
    for side, table, id_col in (("left", data.dataset_1, data.id_column_name_1),
                                ("right", data.dataset_2, data.id_column_name_2)):
        vec = np.load(os.path.join(d, f"{model}_{side}.npy"), mmap_mode="r")
        full_ids = pd.read_csv(os.path.join(BENCH_DIR, name, f"{side}.csv"),
                               usecols=[id_col]).astype(str)[id_col]
        if len(vec) != len(full_ids):
            raise ValueError(
                f"{model}_{side}.npy has {len(vec)} rows but {name}/{side}.csv "
                f"has {len(full_ids)} — embeddings not row-aligned")
        pos = pd.Series(np.arange(len(full_ids)), index=full_ids)
        rows = pos[table[id_col].astype(str)].to_numpy()
        out.append(np.asarray(vec[rows], dtype=np.float32))
    return tuple(out)


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
