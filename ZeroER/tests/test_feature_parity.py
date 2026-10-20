"""``build_zeroer_features`` reproduces genuine Magellan, cell by cell.

The port has to match ``py_entitymatching``'s ``extract_features`` value for
value, and the only way to know is to compare against output that real
``py_entitymatching`` produced. That output exists: ``prepare/reference_features.py`` ran
upstream's own extraction under the py3.6 env over every ladder cell and stored
the matrices in ``datasets/magellan-features/``. **This test reads them.** It does
not re-derive the reference, so it needs neither that env nor upstream's code —
which is the whole reason those matrices were built.

This replaces the deleted ``test_feature_parity.py`` and its 2% sampled fixtures.
Two things are better: the comparison is over *whole* real candidate sets rather
than a sample, and the reference is a stored artefact rather than something
regenerated per run.

It is also the cheap half of what ``test_paper_replication.py`` covers. That test
recomputes features over upstream's much larger original candsets (``abt_buy``
alone is 164,072 pairs) purely to reach an end-to-end F1 — so it verifies the EM,
not the features. Feature correctness is settled here in seconds instead.

One known, characterised difference is asserted rather than tolerated silently:
``dblp_scholar.year_year_lev_sim``. Magellan keeps ``ltable``/``rtable`` separate,
so ``left.year`` stays int64 (``'1995'``) while ``right.year`` — 54% missing — is
float64 (``'1992.0'``); our port takes pyJedAI's single concatenated ``entities``
frame, where ``pd.concat`` of int64+float64 yields float64 and the left side
becomes ``'1995.0'``. It bites only where the two tables carry different numeric
dtypes for one attribute, which among these five benchmarks is that column alone.
The test pins it so a *fix* is noticed as loudly as a regression.

Run:  pytest tests/test_feature_parity.py -v
      RUN_SLOW=1 ... to include amazon_googleproducts (~12 min, see below)
"""
import os
from os.path import exists, join

import numpy as np
import pandas as pd
import pytest

from zeroer.config import BENCH_DIR, FEATURES_DIR, LADDER, PAIRS_DIR, cell_name
from zeroer.data.reference import load_reference_tables
from zeroer.features.strings import build_zeroer_features

# one ULP on a float64 similarity in [0, 1]
TOL = 1e-15

# Magellan resolves each pair through separate tables; we concatenate. Only
# ``dblp_scholar.year`` has mismatched numeric dtypes across the two sides.
KNOWN_DIFFERENCES = {
    "dblp_scholar": {"year_year_lev_sim": 0.3333333333333334},
}

# ``amazon_googleproducts``' left ``description`` reaches 19,263 characters, which
# Magellan types ``str_gt_10w`` and featurises with ``jaccard qgm_3`` / ``cosine
# dlm_dc0`` — q-gram sets over 19k-char strings, per pair. 1,363 pairs costs ~12
# minutes even at ``n_jobs=-1``; every other dataset here is under 35 seconds.
SLOW = {"amazon_googleproducts"}
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                          reason="slow: set RUN_SLOW=1 to run")


def _smallest_cell(dataset):
    """The ladder's cheapest rung. ``LADDER`` is ordered by ascending PQ and PQ
    rises as ``|C|`` falls, so that is the last entry — asserted, so a reorder
    cannot silently point this test at a 64k-pair cell."""
    side, k = LADDER[dataset][-1]
    assert k == min(kk for _, kk in LADDER[dataset]), \
        f"{dataset}: last rung is not the smallest cell — LADDER reordered?"
    return side, k


def _ours(dataset, side, k):
    """Our port's matrix on that cell's stored pairs, row-aligned with them.

    Tables come from ``zeroer.data.reference.load_reference_tables`` rather than being re-read
    here: it applies the same per-dataset prep (``abt_buy``'s description concat,
    ``amazon``'s rename) and the same UTF-8 decoding the stored reference was
    built with. Duplicating that would let the two drift.
    """
    left, right = load_reference_tables(dataset, BENCH_DIR)
    pairs_df = pd.read_csv(join(PAIRS_DIR, cell_name(dataset, side, k) + ".csv"),
                           dtype=str)
    idx_l = {str(v): i for i, v in enumerate(left["id"])}
    idx_r = {str(v): i for i, v in enumerate(right["id"])}
    limit = len(left)
    pairs = [(idx_l[a], limit + idx_r[b])
             for a, b in zip(pairs_df["ltable_id"], pairs_df["rtable_id"])]

    attributes = [c for c in left.columns if c in right.columns]
    return build_zeroer_features(
        pairs, pd.concat([left, right], ignore_index=True), attributes,
        dataset_limit=limit,
        # keep every column: a constant one still has to match, and the drop is
        # candidate-set-dependent so it would make the comparison cell-specific
        drop_zero_variance=False,
        n_jobs=-1)


@pytest.mark.parametrize(
    "dataset",
    [pytest.param(ds, marks=slow) if ds in SLOW else ds for ds in sorted(LADDER)])
def test_matches_stored_magellan_features(dataset):
    side, k = _smallest_cell(dataset)
    ref_path = join(FEATURES_DIR, cell_name(dataset, side, k) + ".csv.gz")
    if not exists(ref_path):
        pytest.skip(f"no stored reference for {cell_name(dataset, side, k)}")

    reference = pd.read_csv(ref_path).drop(columns=["ltable_id", "rtable_id"])
    ours = _ours(dataset, side, k)

    # the column *set* is itself a result: Magellan's type-upgrade rule and
    # name-based filter decide which features exist at all
    assert set(ours.columns) == set(reference.columns), (
        f"column sets differ — only ours: {set(ours.columns) - set(reference.columns)}; "
        f"only Magellan's: {set(reference.columns) - set(ours.columns)}")
    assert len(ours) == len(reference)

    expected = KNOWN_DIFFERENCES.get(dataset, {})
    diffs = {}
    for column in reference.columns:
        worst = np.abs(ours[column].to_numpy(float)
                       - reference[column].to_numpy(float)).max()
        if worst > TOL:
            diffs[column] = worst

    assert set(diffs) == set(expected), (
        f"{dataset}: disagreeing columns changed — got {sorted(diffs)}, "
        f"expected {sorted(expected)}")
    for column, worst in expected.items():
        assert diffs[column] == pytest.approx(worst, rel=1e-6), (
            f"{dataset}.{column}: known difference moved from {worst} to "
            f"{diffs[column]} — characterise it before updating this number")


def test_every_ladder_cell_has_a_stored_reference():
    """Cheap inventory guard: the parity check above is only as good as the
    artefacts, and a half-built tree should fail loudly rather than skip."""
    missing = [cell_name(ds, side, k)
               for ds, cells in LADDER.items() for side, k in cells
               if not exists(join(FEATURES_DIR, cell_name(ds, side, k) + ".csv.gz"))]
    assert not missing, f"{len(missing)} ladder cells lack Magellan features: {missing}"
