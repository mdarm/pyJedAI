"""Generate the Magellan reference feature matrices that ``test_feature_parity``
asserts against. Run manually; the fixtures it writes are committed.

Requires ``py_entitymatching`` 0.4.0, which only installs under Python 3.6 —
hence a standalone script rather than a pytest fixture::

    ~/miniconda3/envs/ZeroER/bin/python tests/generate_magellan_fixtures.py
    ~/miniconda3/envs/ZeroER/bin/python tests/generate_magellan_fixtures.py dblp_acm

The oracle is upstream's *own* code path (``data_loading_helper.feature_extraction
.extract_features`` from a chu-data-lab/zeroer checkout), not a reimplementation
of it, so the fixtures are genuine py_entitymatching output.

Both sides are fed **this repo's** tables, read as UTF-8. That matters: our CSVs
are UTF-8 re-encodings of upstream's ISO-8859-1 ones, and on ``dblp_scholar``
they are not even logically equal (upstream's ``scholar.csv`` is doubly
mojibaked, ours has one layer decoded). Pointing Magellan at upstream's copies
would therefore compare two different inputs and report a feature-code failure
that is really a data-provenance difference. Comparing on identical input is
what isolates ``zeroer_features``.

Candidate pairs are sampled from upstream's blocked candsets — the pair
distribution ZeroER actually runs on, including the missing-value patterns
blocking lets through.
"""
import os
import sys
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd

HERE = dirname(abspath(__file__))
FIXTURES = join(HERE, "fixtures", "magellan")

# A chu-data-lab/zeroer checkout: supplies both the upstream feature-extraction
# code and the blocked candsets we sample from.
UPSTREAM = os.environ.get(
    "ZEROER_UPSTREAM", join(dirname(dirname(dirname(HERE))), "zeroer"))

SEED = 0
SAMPLE_FRAC = 0.02      # of each blocked candset
MIN_PAIRS = 300         # floor, so fodors_zagats' 2.9k candset still gets cover
MAX_PAIRS = 3000        # ceiling, to bound committed fixture size

# Upstream's table prep, reproduced per dataset:
#   rename   — column renames applied before featurisation
#   concat   — (target, source) columns joined with a space on the right table;
#              upstream's ``block_abt_buy`` mutates B this way before extraction
DATASETS = {
    "fodors_zagats": {},
    "dblp_acm": {},
    "dblp_scholar": {},
    "abt_buy": {"concat": ("description", "manufacturer")},
    "amazon_googleproducts": {"rename": {"name": "title"}},
}


def _tables(dataset: str, bench_dir: str):
    """This repo's two tables, UTF-8, with upstream's prep applied."""
    spec = DATASETS[dataset]
    left = pd.read_csv(join(bench_dir, dataset, "left.csv"), encoding="utf-8")
    right = pd.read_csv(join(bench_dir, dataset, "right.csv"), encoding="utf-8")
    if "rename" in spec:
        right = right.rename(columns=spec["rename"])
    if "concat" in spec:
        target, source = spec["concat"]
        right[target] = right[target] + " " + right[source]
    return left, right


def _sample_pairs(dataset: str, rng: np.random.RandomState) -> pd.DataFrame:
    """Seeded sample of upstream's blocked candset for one dataset."""
    path = join(UPSTREAM, "datasets", dataset, "candset_features_df.csv")
    if not exists(path):
        raise FileNotFoundError(
            f"{path} not found — run upstream ZeroER on {dataset} first, or set "
            "ZEROER_UPSTREAM to a checkout that has")
    candset = pd.read_csv(path, usecols=["ltable_id", "rtable_id"],
                          dtype={"ltable_id": str, "rtable_id": str})
    n = int(np.clip(round(len(candset) * SAMPLE_FRAC), MIN_PAIRS, MAX_PAIRS))
    n = min(n, len(candset))
    rows = rng.choice(len(candset), size=n, replace=False)
    return candset.iloc[np.sort(rows)].reset_index(drop=True)


def generate(dataset: str, bench_dir: str) -> str:
    from data_loading_helper.feature_extraction import extract_features
    import py_entitymatching as em

    rng = np.random.RandomState(SEED)
    pairs = _sample_pairs(dataset, rng)
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

    os.makedirs(FIXTURES, exist_ok=True)
    out = join(FIXTURES, f"{dataset}.csv.gz")
    features.to_csv(out, index=False, compression="gzip")
    print("wrote {} — {} pairs x {} features".format(
        out, len(features), features.shape[1] - 2))
    return out


if __name__ == "__main__":
    sys.path.insert(0, UPSTREAM)
    # ``data.BENCH_DIR`` is the same path, but data.py is py3.7+ syntax and this
    # script runs on 3.6 — so resolve it directly rather than importing.
    bench_dir = join(dirname(HERE), "datasets")

    for name in (sys.argv[1:] or list(DATASETS)):
        print("=== {} ===".format(name))
        generate(name, bench_dir)
