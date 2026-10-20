"""Build genuine Magellan features on the frozen candidate sets.

Run from the repository root, using the matching interpreter for each stage:

    ../../data-env/bin/python -m prepare.reference_features pairs
    ~/micromamba/envs/ZeroER/bin/python -m prepare.reference_features features

Pairs use modern pyJedAI/FAISS; features use upstream py_entitymatching under
Python 3.6. Both stages read this repository's UTF-8 tables.
"""
import os
import sys
from os.path import exists, join

import numpy as np
import pandas as pd

from zeroer.config import (BENCH_DIR, DISTANCE, FEATURES_DIR, LADDER, MODEL, PAIRS_DIR,
                    UPSTREAM, cell_name)
from zeroer.data.reference import load_reference_tables

def export_pairs(dataset):
    """Write one ``(ltable_id, rtable_id)`` CSV per ladder cell."""
    from zeroer.data.benchmarks import load_data, load_embeddings
    from zeroer.estimator import nn_search

    data = load_data(dataset)
    v1, v2 = load_embeddings(dataset, MODEL, data)
    left_ids = data.dataset_1[data.id_column_name_1].astype(str).to_numpy()
    right_ids = data.dataset_2[data.id_column_name_2].astype(str).to_numpy()

    os.makedirs(PAIRS_DIR, exist_ok=True)
    written = []
    for side, k in LADDER[dataset]:
        # nn_search indexes its first argument and queries its second, so the
        # queried side is the second one — the sweep's own wiring.
        index_side, query_side = (v1, v2) if side == "right" else (v2, v1)
        nbrs = nn_search(index_side, query_side, k, DISTANCE)
        queried = np.repeat(np.arange(len(nbrs)), k)
        if side == "right":
            li, ri = nbrs.ravel(), queried       # neighbours are left rows
        else:
            li, ri = queried, nbrs.ravel()       # neighbours are right rows

        out = join(PAIRS_DIR, cell_name(dataset, side, k) + ".csv")
        pd.DataFrame({"ltable_id": left_ids[li],
                      "rtable_id": right_ids[ri]}).to_csv(out, index=False)
        written.append((os.path.basename(out), len(li)))
        print("wrote {} — {} pairs".format(os.path.basename(out), len(li)))
    return written


def extract_pair_features(dataset, pairs, bench_dir=BENCH_DIR, n_jobs=None):
    """Genuine upstream features for supplied pairs, without writing a cache.

    ``n_jobs`` optionally caps extraction workers without changing feature math.
    This lets cache assembly compute only pairs absent from existing artifacts.
    """
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    from data_loading_helper.feature_extraction import extract_features
    import py_entitymatching as em

    pairs = pairs[["ltable_id", "rtable_id"]].astype(str).reset_index(drop=True)
    left, right = load_reference_tables(dataset, bench_dir)

    # Magellan resolves each pair's values through the fk columns, so the
    # candset only needs its key and the two foreign keys — plus ``gold``,
    # which upstream's ``extract_features`` passes as ``attrs_after``.
    candset = pd.DataFrame({
        "_id": np.arange(len(pairs)),
        "ltable_id": pairs["ltable_id"].values,
        "rtable_id": pairs["rtable_id"].values,
        "gold": 0,
    })
    for table in (left, right):
        table["id"] = table["id"].astype(str)
    em.set_key(left, "id")
    em.set_key(right, "id")
    em.set_key(candset, "_id")
    em.set_property(candset, "ltable", left)
    em.set_property(candset, "rtable", right)
    em.set_property(candset, "fk_ltable", "ltable_id")
    em.set_property(candset, "fk_rtable", "rtable_id")

    original = em.extract_feature_vecs
    if n_jobs is not None:
        def limited_workers(*args, **kwargs):
            kwargs["n_jobs"] = n_jobs
            return original(*args, **kwargs)
        em.extract_feature_vecs = limited_workers
    try:
        features = extract_features(left, right, candset)
    finally:
        em.extract_feature_vecs = original
    return features.drop(columns=["_id", "gold"])


def build_features(dataset, side, k, bench_dir=BENCH_DIR):
    """Run upstream's ``extract_features`` on one cell's pairs."""
    name = cell_name(dataset, side, k)
    pairs_path = join(PAIRS_DIR, name + ".csv")
    if not exists(pairs_path):
        raise FileNotFoundError(
            "{} not found — run `python -m prepare.reference_features pairs` under data-env "
            "first".format(pairs_path))
    pairs = pd.read_csv(pairs_path, dtype={"ltable_id": str, "rtable_id": str})
    features = extract_pair_features(dataset, pairs, bench_dir=bench_dir)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out = join(FEATURES_DIR, name + ".csv.gz")
    features.to_csv(out, index=False, compression="gzip")
    print("wrote {} — {} pairs x {} features".format(
        os.path.basename(out), len(features), features.shape[1] - 2))
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2:] or sorted(LADDER)
    if stage == "pairs":
        for dataset in only:
            export_pairs(dataset)
    elif stage == "features":
        for dataset in only:
            for side, k in LADDER[dataset]:
                build_features(dataset, side, k)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
