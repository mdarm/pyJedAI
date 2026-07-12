"""End-to-end replication of the original ZeroER results from this port.

Each test replays the upstream pipeline (chu-data-lab/zeroer) using this
repo's components: the *original* Magellan candidate sets (pinned under
``tests/fixtures/`` — Magellan's blocking cannot run in this environment)
-> exact feature generation (``zeroer_features``) -> stock EM
(``module.ZeroerModel``, c_bay=0.1, max_iter=40, as upstream ``run_zeroer``).

The asserted F1s are what the upstream code itself produces on these candidate
sets (2-decimal rounding, as upstream prints). Where the repo's blocking is
faithful to the paper's setup they equal Table 3 of the paper: fodors_zagats
with transitivity = 1.00 and dblp_acm = 0.96. abt_buy scores 0.16 against the
paper's 0.52 because the repo ships a much looser blocking function than the
paper's LSH blocker — the 0.16 is still the exact upstream-code result.

Feature generation is pure-Python string similarity over every candidate
pair, so the large candidate sets take a while: only fodors_zagats runs by
default; set RUN_SLOW=1 to include the rest.

Run:  pytest tests/ -v            (fodors_zagats only, ~1 min)
      RUN_SLOW=1 pytest tests/ -v (all datasets, tens of minutes)
"""
import os
import sys
from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))  # ZeroER/

from module import ZeroerModel, get_y_init_given_threshold
from utils import BENCH_DIR
from zeroer_features import build_zeroer_features

FIXTURES = join(dirname(os.path.abspath(__file__)), "fixtures")

# expected_f1: upstream-code result on the pinned candidate set (this is what
# the port must reproduce exactly); paper_f1: the paper's Table 3 number, for
# reference where blocking differences make the two diverge.
CASES = {
    "fodors_zagats": dict(expected_f1=0.98, paper_f1=1.00),
    "dblp_acm": dict(expected_f1=0.96, paper_f1=0.96),
    "abt_buy": dict(expected_f1=0.16, paper_f1=0.52),
}

SLOW = {"dblp_acm", "abt_buy"}
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                          reason="slow: set RUN_SLOW=1 to run")


def _load_case(dataset):
    """(pairs, entities, attributes, dataset_limit, y_true) for one benchmark,
    prepared exactly as the upstream pipeline saw it."""
    d = join(BENCH_DIR, dataset)
    lt = pd.read_csv(join(d, "left.csv"), encoding="iso-8859-1")
    rt = pd.read_csv(join(d, "right.csv"), encoding="iso-8859-1")
    if dataset == "amazon_googleproducts":   # upstream table prep
        rt = rt.rename(columns={"name": "title"})
    if dataset == "abt_buy":                  # upstream block_abt_buy mutates B
        rt["description"] = rt["description"] + " " + rt["manufacturer"]

    candset = pd.read_csv(join(FIXTURES, f"{dataset}_candset.csv.gz"))
    idx_l = {str(v): i for i, v in enumerate(lt["id"])}
    idx_r = {str(v): i for i, v in enumerate(rt["id"])}
    limit = len(lt)
    pairs = [(idx_l[str(a)], limit + idx_r[str(b)])
             for a, b in zip(candset["ltable_id"], candset["rtable_id"])]

    gt = pd.read_csv(join(d, "matches.csv"))
    gold = set(zip(gt.iloc[:, 0].astype(str), gt.iloc[:, 1].astype(str)))
    y_true = np.array([(str(a), str(b)) in gold
                       for a, b in zip(candset["ltable_id"],
                                       candset["rtable_id"])], dtype=int)

    entities = pd.concat([lt, rt], ignore_index=True)
    attributes = [c for c in lt.columns if c in rt.columns]  # all shared attrs
    return pairs, entities, attributes, limit, y_true


def _run_em(features, y_true, run_trans=False, LR_dup_free=False, id_df=None):
    y_init = get_y_init_given_threshold(features)
    _, p_m = ZeroerModel.run_em(
        (features.values, None, None), features.columns, (y_init, None, None),
        (id_df, None, None), LR_dup_free, False, run_trans,
        c_bay=0.1, y_true=y_true, max_iter=40)   # as upstream run_zeroer
    return f1_score(y_true, np.round(np.clip(p_m + 1e-300, 0., 1.)).astype(int))


@pytest.mark.parametrize(
    "dataset",
    [pytest.param(ds, marks=slow) if ds in SLOW else ds for ds in CASES])
def test_replicates_original_zeroer(dataset):
    pairs, entities, attributes, limit, y_true = _load_case(dataset)
    features = build_zeroer_features(pairs, entities, attributes,
                                     dataset_limit=limit)
    f1 = _run_em(features, y_true)
    assert round(f1, 2) == CASES[dataset]["expected_f1"]


def test_fodors_zagats_transitivity_matches_paper():
    """Full ZeroER (transitivity, dup-free tables) hits the paper's 1.00."""
    pairs, entities, attributes, limit, y_true = _load_case("fodors_zagats")
    features = build_zeroer_features(pairs, entities, attributes,
                                     dataset_limit=limit)
    id_df = pd.DataFrame(pairs, columns=["ltable_id", "rtable_id"])
    f1 = _run_em(features, y_true, run_trans=True, LR_dup_free=True,
                 id_df=id_df)
    assert round(f1, 2) == 1.00
