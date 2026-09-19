"""Genuine py_entitymatching feature matrices on the ladder's candidate sets.

Two stages, two interpreters — ``py_entitymatching`` 0.4.0 only installs under
Python 3.6, and faiss/pyjedai only exist in ``data-env``::

    data-env/bin/python reference_features.py pairs
    micromamba/envs/ZeroER/bin/python reference_features.py features

Stage 1 re-runs the blocking sweep's exact k-NN for each ladder cell and writes
the ``(ltable_id, rtable_id)`` pairs. Stage 2 feeds those pairs to upstream's
own ``data_loading_helper.feature_extraction.extract_features`` — the real
Magellan path, not a reimplementation — and stores the result.

The output is the *reference* string-feature matrix for the matching
experiments: same pairs the matcher scores, features computed by upstream's
code rather than by this repo's ``zeroer_features`` port.

Both sides are fed **this repo's** tables, read as UTF-8. That matters: our CSVs
are UTF-8 re-encodings of upstream's ISO-8859-1 ones, and on ``dblp_scholar``
they are not even logically equal (upstream's ``scholar.csv`` is doubly
mojibaked, ours has one layer decoded).
"""
import os
import sys
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd

HERE = dirname(abspath(__file__))
# Spelled out rather than imported from ``data``: that module needs pyjedai and
# ``from __future__ import annotations``, neither available under Python 3.6.
BENCH_DIR = join(HERE, "datasets")
PAIRS_DIR = join(HERE, "outputs", "ladder_pairs")
FEATURES_DIR = join(HERE, "datasets", "magellan-features")

# A chu-data-lab/zeroer checkout, for the upstream feature-extraction code.
UPSTREAM = os.environ.get("ZEROER_UPSTREAM", join(dirname(dirname(HERE)), "zeroer"))

# The blocking model the ladder is read off (reporting.efficiency_leaderboard's
# top model), and the distance the sweep used throughout.
MODEL = "tencent_KaLM-Embedding-Gemma3-12B-2511"
DISTANCE = "cosine"

# Ladder cells, per dataset: the (queried_side, top_k) nearest each PQ rung in
# {0.08, 0.2, 0.4, 0.8}, read off the `<dataset>_blocking` studies. fodors_zagats
# tops out at PQ 0.338, so it has three rungs rather than four.
LADDER = {
    "fodors_zagats":         [("right", 4), ("left", 1), ("right", 1)],
    "dblp_acm":              [("right", 12), ("right", 5), ("left", 2), ("left", 1)],
    "abt_buy":               [("left", 13), ("right", 5), ("left", 2), ("left", 1)],
    "amazon_googleproducts": [("right", 5), ("right", 2), ("left", 2), ("left", 1)],
    "dblp_scholar":          [("right", 1), ("left", 10), ("left", 5), ("left", 1)],
}

# Upstream's table prep, reproduced per dataset:
#   rename — column renames applied before featurisation
#   concat — (target, source) columns joined with a space on the right table;
#            upstream's ``block_abt_buy`` mutates B this way before extraction
PREP = {
    "fodors_zagats": {},
    "dblp_acm": {},
    "dblp_scholar": {},
    "abt_buy": {"concat": ("description", "manufacturer")},
    "amazon_googleproducts": {"rename": {"name": "title"}},
}


def cell_name(dataset, side, k):
    return "{}__{}_k{}".format(dataset, side, k)


def _tables(dataset, bench_dir):
    """This repo's two tables, UTF-8, with upstream's prep applied."""
    spec = PREP[dataset]
    left = pd.read_csv(join(bench_dir, dataset, "left.csv"), encoding="utf-8")
    right = pd.read_csv(join(bench_dir, dataset, "right.csv"), encoding="utf-8")
    if "rename" in spec:
        right = right.rename(columns=spec["rename"])
    if "concat" in spec:
        target, source = spec["concat"]
        right[target] = right[target] + " " + right[source]
    return left, right


# --------------------------------------------------------------------------
# Stage 1 — data-env: the ladder's candidate pairs
# --------------------------------------------------------------------------

def export_pairs(dataset):
    """Write one ``(ltable_id, rtable_id)`` CSV per ladder cell."""
    from data import load_data, load_embeddings
    from estimator import nn_search

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


# --------------------------------------------------------------------------
# Stage 2 — ZeroER env: upstream's Magellan features on those pairs
# --------------------------------------------------------------------------

def build_features(dataset, side, k, bench_dir=BENCH_DIR):
    """Run upstream's ``extract_features`` on one cell's pairs."""
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    from data_loading_helper.feature_extraction import extract_features
    import py_entitymatching as em

    name = cell_name(dataset, side, k)
    pairs_path = join(PAIRS_DIR, name + ".csv")
    if not exists(pairs_path):
        raise FileNotFoundError(
            "{} not found — run `reference_features.py pairs` under data-env "
            "first".format(pairs_path))
    pairs = pd.read_csv(pairs_path, dtype={"ltable_id": str, "rtable_id": str})
    left, right = _tables(dataset, bench_dir)

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

    features = extract_features(left, right, candset)
    features = features.drop(columns=["_id", "gold"])

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
