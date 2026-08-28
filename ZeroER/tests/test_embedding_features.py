"""``embedding_features`` obeys the conventions ZeroER's EM depends on.

Four invariants, each one a way the feature matrix can be wrong while looking
right: the column *names* decide the covariance blocks, an absent value must read
as exactly 0, the vectors must follow ``Data`` when it is a subsampled tuning
partition, and ``BlockFeatureCache``'s ``top_k`` slice must stay exact once extra
columns ride along with the string ones.

Needs the per-attribute embedding tree, which is gitignored (64 GB); the tests
skip when it is absent. ``minilm6`` is used throughout — 384 dimensions keeps
this fast, and nothing here is model-specific.

Run:  pytest tests/test_embedding_features.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

from data import ATTR_DIR, DATASETS, RECORD_DIR, load_data, load_embeddings
from embedding_features import build_embedding_features, embedding_feature_builder
from estimator import BlockFeatureCache, ZeroEREstimator, nn_search

MODEL = "minilm6"
CLEAN = "fodors_zagats"        # no missing values anywhere
SPARSE = "abt_buy"            # left `price` 61% absent, right `description` 40%


def _require(dataset):
    """Skip unless both trees carry ``MODEL`` for this dataset."""
    needed = [os.path.join(RECORD_DIR, dataset, f"{MODEL}_left.npy")]
    needed += [os.path.join(ATTR_DIR, dataset, f"{MODEL}_{a}_{side}.npy")
               for a in DATASETS[dataset]["attributes"] for side in ("left", "right")]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        pytest.skip(f"no {MODEL} embeddings for {dataset}: {missing[0]} absent")


def _knn_pairs(dataset, data, k):
    """Right-queried k-NN candidate pairs, the form the cache builds."""
    v1, v2 = load_embeddings(dataset, MODEL, data)
    nbrs = nn_search(v1, v2, k, "cosine")
    return np.column_stack([nbrs.ravel(),
                            data.dataset_limit
                            + np.repeat(np.arange(len(nbrs)), nbrs.shape[1])])


def test_columns_are_named_into_their_attribute_block():
    """``module.py`` groups by ``name.split("_")[0]``, so the name *is* the block."""
    _require(CLEAN)
    data = load_data(CLEAN)
    attrs = DATASETS[CLEAN]["attributes"]
    features = build_embedding_features(
        _knn_pairs(CLEAN, data, 2), CLEAN, MODEL, attrs, data,
        metrics=("cos", "linf"))

    assert list(features.columns) == [f"{a}_emb_{m}"
                                      for a in attrs for m in ("cos", "linf")]
    for column in features.columns:
        assert column.split("_")[0] in attrs, (
            f"{column} would open its own singleton covariance group")


def test_present_rows_are_the_cosine_and_absent_rows_are_zero():
    _require(SPARSE)
    data = load_data(SPARSE)
    attrs = DATASETS[SPARSE]["attributes"]
    pairs = _knn_pairs(SPARSE, data, 2)
    features = build_embedding_features(pairs, SPARSE, MODEL, attrs, data)

    li, ri = pairs[:, 0], pairs[:, 1] - data.dataset_limit
    saw_absent = False
    for attr in attrs:
        left, right = load_embeddings(SPARSE, f"{MODEL}_{attr}", data, base=ATTR_DIR)
        u, v = left[li], right[ri]
        absent = ~(u.any(axis=1) & v.any(axis=1))
        saw_absent |= bool(absent.any())
        got = features[f"{attr}_emb_cos"].to_numpy()

        assert (got[absent] == 0.0).all(), f"{attr}: absent pair carries a value"
        np.testing.assert_allclose(
            got[~absent], np.einsum("ij,ij->i", u[~absent], v[~absent]),
            rtol=0, atol=1e-6)
    assert saw_absent, f"{SPARSE} was chosen because it has absent values"


def test_partition_follows_the_subsampled_tables():
    """``presence.npz`` is indexed by full-table rows; the vectors' zero rows are
    not, and re-selection by id has to keep the two in step."""
    _require(SPARSE)
    full, part = load_data(SPARSE), load_data(SPARSE, partition=True)
    attrs = DATASETS[SPARSE]["attributes"]
    assert len(part.dataset_1) < len(full.dataset_1), "partition did not subsample"

    # the same 20 entity pairs, addressed in each frame's own row space
    row_of = lambda table, col: {v: i for i, v in enumerate(table[col].astype(str))}
    p_left, p_right = row_of(part.dataset_1, "id"), row_of(part.dataset_2, "id")
    f_left, f_right = row_of(full.dataset_1, "id"), row_of(full.dataset_2, "id")
    ids = [(l, r) for l, r in zip(list(p_left)[:20], list(p_right)[:20])]

    got = build_embedding_features(
        np.array([(p_left[l], part.dataset_limit + p_right[r]) for l, r in ids]),
        SPARSE, MODEL, attrs, part)
    expected = build_embedding_features(
        np.array([(f_left[l], full.dataset_limit + f_right[r]) for l, r in ids]),
        SPARSE, MODEL, attrs, full)

    pd.testing.assert_frame_equal(got, expected)


def test_cache_slice_stays_exact_with_extra_columns():
    """``BlockFeatureCache`` builds at ``k_max`` and serves ``top_k`` as a slice."""
    _require(CLEAN)
    data = load_data(CLEAN)
    attrs = DATASETS[CLEAN]["attributes"]
    builder = embedding_feature_builder(CLEAN, MODEL, attrs, data)

    cache = BlockFeatureCache(CLEAN, data, attrs, k_max=5, n_jobs=1,
                              extra_features=builder)
    pairs, features, _ = cache.get(MODEL, "cosine", 2)

    direct_pairs = _knn_pairs(CLEAN, data, 2)
    np.testing.assert_array_equal(pairs, direct_pairs)

    expected = build_embedding_features(direct_pairs, CLEAN, MODEL, attrs, data)
    served = [c for c in expected.columns if c in features.columns]
    assert served, "no embedding column survived the constant-column drop"
    for column in served:
        np.testing.assert_allclose(features[column].to_numpy(),
                                   expected[column].to_numpy(), rtol=0, atol=0)


def test_estimator_refuses_cache_and_extra_features_together():
    """On the cache path the cache owns feature generation, so the estimator's
    own hook would be a silent no-op."""
    _require(CLEAN)
    data = load_data(CLEAN)
    attrs = DATASETS[CLEAN]["attributes"]
    cache = BlockFeatureCache(CLEAN, data, attrs, k_max=2, n_jobs=1)

    with pytest.raises(ValueError, match="pass extra_features to the BlockFeatureCache"):
        ZeroEREstimator(blocker="precomputed", cache=cache,
                        extra_features=lambda pairs: pd.DataFrame())


def test_unknown_metric_is_rejected():
    _require(CLEAN)
    data = load_data(CLEAN)
    with pytest.raises(KeyError, match="unknown metric"):
        build_embedding_features(_knn_pairs(CLEAN, data, 1), CLEAN, MODEL,
                                 DATASETS[CLEAN]["attributes"], data,
                                 metrics=("cos", "l2"))
