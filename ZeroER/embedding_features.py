"""Per-attribute embedding similarity features for the ZeroER matcher (Arm B2).

DeepER's *averaging* form: one similarity per attribute, computed between the two
records' attribute vectors, so the output is an *m*-dimensional similarity vector
whose shape is compatible with ZeroER's covariance decomposition
(``deeper-distributed-representations``; ``zeroer-entity-resolution-zero-labels``
Innovation 1). Whole-record similarity is deliberately *not* offered here — a
single scalar summarising the pair correlates with every attribute group, and
``module.py`` would file it into a singleton group, forcing an independence that
does not hold.

Columns are named ``{attr}_emb_{metric}``, so ``module.py``'s
``name.split("_")[0]`` files each one into that attribute's covariance block. The
name is a modelling decision, not a label.

Three conventions this module has to keep, all load-bearing:

* **Orientation.** ``module.get_y_init_given_threshold`` min-max scales every
  column, sums them and thresholds at 0.8, so every feature must be *increasing*
  in match likelihood. Distances are therefore mapped through ``1/(1+d)``; a raw
  distance would push the EM initialisation the wrong way.
* **Absent values -> 0**, read off the vectors themselves. The encoders leave an
  absent value as an exact zero row (``datasets/embeddings/README.md``), so
  ``~vec.any(axis=1)`` recovers the mask *after* ``load_embeddings`` has
  re-selected rows by id. The sibling ``presence.npz`` is indexed by full-table
  rows and would silently mis-index a tuning partition, so it is not used here.
  Forcing the feature to 0 when either side is absent is exactly
  ``zeroer_features``' Magellan convention (NaN in either value -> NaN feature ->
  0 after ``fillna``).
* **Pair-local values only.** ``estimator.BlockFeatureCache`` builds once at
  ``k_max`` and serves smaller ``top_k`` as a row slice, which is exact only for
  features that depend on nothing but their own pair. A blocking-rank or
  margin-to-next-neighbour feature would break that invariant and must not be
  added to the pair-local builders. ``fuse`` runs later on each fitted cell.

**On the metric set.** For L2-normalised vectors ``‖u-v‖₂² = 2-2cos``, so a
Euclidean-derived similarity is a strictly monotone function of the cosine and
adds no information: measured Spearman against ``cos`` is exactly +1.0000, and
``1/(1+‖u-v‖₁)`` reaches +0.9994 while collapsing into a near-constant range
(0.018-0.071 on ``abt_buy``'s ``description``) — i.e. straight into the
singularity condition Innovation 2 exists to regularise. Only ``linf`` carries
meaningfully distinct rank information (Spearman 0.86-0.94). ``cos`` is therefore
the default and ``linf`` the one alternative worth a column.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from config import ATTR_DIR
from data import FAMILIES, load_embeddings

__all__ = ["METRICS", "FUSIONS", "build_embedding_features", "embedding_feature_builder", "fuse"]


def _cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine, i.e. the plain inner product — the vectors are already L2-normalised."""
    return np.einsum("ij,ij->i", u, v)


def _linf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``1/(1+‖u-v‖∞)``: the largest single-dimension disagreement, oriented."""
    return 1.0 / (1.0 + np.abs(u - v).max(axis=1))


# Extension point. A new entry must be increasing in match likelihood and depend
# on its own pair only; see the module docstring on both counts.
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cos": _cos,
    "linf": _linf,
}


def build_embedding_features(
    pairs,
    dataset: str,
    model: str,
    attributes: Sequence[str],
    data,
    metrics: Iterable[str] = ("cos",),
    base: str = ATTR_DIR,
) -> pd.DataFrame:
    """``{attr}_emb_{metric}`` columns for ``pairs``, rows aligned with ``pairs``.

    pairs: (left_row, right_row) positional indices into ``data.entities``, right
        rows offset by ``data.dataset_limit`` — the same form
        ``zeroer_features.build_zeroer_features`` takes, so the two matrices
        concatenate directly.
    dataset, model: keys into ``datasets/embeddings/attribute/<dataset>/``; the
        per-attribute file for ``(model, attr, side)`` must exist.
    attributes: attributes to featurise, normally ``data.DATASETS[ds]["attributes"]``.
    data: the pyJedAI ``Data`` the pairs index into. Vectors are re-selected by
        id against it, so a subsampled tuning partition works unchanged.
    metrics: names from ``METRICS``. One column per (attribute, metric).

    Vectors are loaded and released one attribute at a time — ``dblp_scholar``'s
    right table is ~1 GB per attribute per model at 3840 dimensions. Amortise
    repeated calls with ``estimator.BlockFeatureCache``, not by holding them here.
    """
    if getattr(data, "is_dirty_er", False):
        raise ValueError("per-attribute embedding features support clean-clean ER only")
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; have {sorted(METRICS)}")
    pairs = np.asarray(pairs)
    if len(pairs) == 0:
        return pd.DataFrame()

    limit = data.dataset_limit
    li, ri = pairs[:, 0], pairs[:, 1] - limit

    columns = {}
    for attr in attributes:
        left, right = load_embeddings(dataset, f"{model}_{attr}", data, base=base)
        u, v = left[li], right[ri]
        # an exact zero row is the encoders' "value absent" marker
        present = u.any(axis=1) & v.any(axis=1)
        for name in metrics:
            columns[f"{attr}_emb_{name}"] = np.where(present, METRICS[name](u, v), 0.0)
        del left, right, u, v
    return pd.DataFrame(columns)


def embedding_feature_builder(
    dataset: str,
    model: str,
    attributes: Sequence[str],
    data,
    metrics: Iterable[str] = ("cos",),
    base: str = ATTR_DIR,
) -> Callable[[object], pd.DataFrame]:
    """``build_embedding_features`` with everything but ``pairs`` bound.

    The shape ``estimator.BlockFeatureCache(extra_features=...)`` and
    ``ZeroEREstimator(extra_features=...)`` expect, so the estimator stays
    ignorant of the on-disk embedding layout the same way ``vectors=`` keeps it
    ignorant for blocking.
    """
    metrics = tuple(metrics)

    def build(pairs) -> pd.DataFrame:
        return build_embedding_features(pairs, dataset, model, attributes, data,
                                       metrics=metrics, base=base)

    build.__doc__ = (f"{list(attributes)} x {list(metrics)} embedding features "
                     f"from {model} on {dataset}")
    return build


FUSIONS = ("separate", "mean", "zmean", "family")


def fuse(bank, models, attributes, present, fusion="separate", metric="cos"):
    """The embedding feature columns of one *evaluated* cell, from its bank rows.

    ``bank``: ``load_bank`` rows of exactly the pairs being fitted; ``present``:
    {attribute: boolean array}. One column group per attribute, named into that
    attribute's block:

    * ``separate`` — one column per model; the EM sees every contrast.
    * ``mean`` — the raw mean of the cosines: for L2-normalised vectors this *is*
      the cosine of the concatenated embeddings.
    * ``zmean`` — the mean of the per-model z-scored cosines.
    * ``family`` — one ``zmean`` per encoder family present in ``models``.

    z-scales are fitted here, on the fitted pairs' present rows — never once on a
    larger cell and reused. A fused z-column is mapped back onto [0, 1] over the
    present rows, with 0 where the attribute is absent: upstream's convention for
    every similarity feature, so the initialisation reads it the same way.
    """
    if fusion not in FUSIONS:
        raise ValueError("fusion must be one of {}".format(FUSIONS))
    out = {}
    for attr in attributes:
        there = np.asarray(present[attr], dtype=bool)
        columns = bank[["{}_emb_{}__{}".format(attr, metric, m) for m in models]]
        if fusion == "separate":
            out.update({c: columns[c].to_numpy() for c in columns})
            continue
        if fusion == "mean":
            out["{}_emb_{}__mean".format(attr, metric)] = columns.mean(axis=1).to_numpy()
            continue
        z = (columns - columns[there].mean()) / columns[there].std().replace(0, 1)
        z.columns = models
        groups = {"zmean": models} if fusion == "zmean" else \
            {family: [m for m in members if m in models] for family, members in FAMILIES.items()}
        for name, members in groups.items():
            if members:
                score = z[members].mean(axis=1).to_numpy()
                low, high = score[there].min(), score[there].max()
                out["{}_emb_{}__{}".format(attr, metric, name)] = \
                    np.where(there, (score - low) / ((high - low) or 1.0), 0.0)
    return pd.DataFrame(out)
