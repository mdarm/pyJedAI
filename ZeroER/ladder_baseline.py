"""String-only ZeroER on the 19 ladder cells, under the two baseline configurations.

    data-env/bin/python ladder_baseline.py              # -> outputs/ladder_baseline.csv
    data-env/bin/python ladder_baseline.py sensitivity  # -> outputs/ladder_sensitivity.csv
    data-env/bin/python ladder_baseline.py transitivity # -> outputs/ladder_transitivity.csv

``B-paper`` is the paper's stated configuration (1908.06049v2 sections 3.3, 5.1):
κ′ = 0.01, initialisation threshold ε = 0.5, stop when the expected
log-likelihood moves < 1e-5, at most 200 iterations. ``B-code`` is what the
released code runs: ``c_bay`` = 0.1, threshold 0.8, its own free-energy stop, 40
iterations. As in the paper, ZeroER is fitted on the whole candidate set — its
50/50 train/test split is for the supervised baselines only.

No k-NN and no feature generation: each cell is its stored pairs and genuine
Magellan matrix (``reference_features.py``). P/R/F1 are against the *full* gold
standard, so matches the blocker missed count against recall. ``accept_all_f1``
(declare every candidate a match) and ``oracle_f1`` (a perfect matcher on these
candidates) are the floor and ceiling the matcher has to be read between.
"""
import os
import sys
import time
from os.path import join

import pandas as pd

from matcher import ZeroERMatcher
from reference_features import BENCH_DIR, FEATURES_DIR, LADDER, cell_name
from tuning import OUTPUT_DIR

CONFIGS = {
    "B-paper": dict(c_bay=0.01, init_threshold=0.5, tol=1e-5, max_iter=200),
    "B-code": dict(c_bay=0.1, init_threshold=0.8, tol=None, max_iter=40),
}

# The paper calls ZeroER "fairly robust for κ′ in [0, 0.1]" and its initialisation
# robust to ε (section 5.4). One knob at a time around B-paper, on the three
# datasets that are not saturated, to see whether that holds on k-NN candidates.
SENSITIVITY_DATASETS = ["abt_buy", "amazon_googleproducts", "dblp_scholar"]
C_BAYS = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3]
INIT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def sensitivity_configs():
    paper = CONFIGS["B-paper"]
    configs = {"c_bay={}".format(c): dict(paper, c_bay=c) for c in C_BAYS}
    configs.update({"init_threshold={}".format(e): dict(paper, init_threshold=e)
                    for e in INIT_THRESHOLDS if e != paper["init_threshold"]})
    return configs


# In-model transitivity assumes duplicate-free tables. That holds on
# fodors_zagats and dblp_acm (gold degree 1/1); on abt_buy (2/2) and
# amazon_googleproducts (5/2) it is a constraint ablation, not an exact
# assumption; dblp_scholar (20/4) is excluded — the paper says it is wrong there.
TRANSITIVITY_DATASETS = ["fodors_zagats", "dblp_acm", "abt_buy", "amazon_googleproducts"]

# Column groups in the stored matrices that are outside data.DATASETS' attribute
# allow-list: fodors_zagats' 'class' is a ground-truth leak, 'type' goes with it.
DROP_GROUPS = {"fodors_zagats": ["type", "class"]}


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def load_cell(dataset, side, k):
    """(pairs, features, is_gold, n_gold) for one ladder cell."""
    stored = pd.read_csv(join(FEATURES_DIR, cell_name(dataset, side, k) + ".csv.gz"),
                         dtype={"ltable_id": str, "rtable_id": str})
    gt = pd.read_csv(join(BENCH_DIR, dataset, "matches.csv"), dtype=str)
    gold = set(zip(gt.iloc[:, 0], gt.iloc[:, 1]))
    is_gold = pd.Series([pair in gold for pair in
                         zip(stored["ltable_id"], stored["rtable_id"])]).to_numpy()

    features = stored.drop(columns=["ltable_id", "rtable_id"])
    group = features.columns.str.split("_").str[0]
    features = features.loc[:, ~group.isin(DROP_GROUPS.get(dataset, []))]
    features = features.loc[:, features.nunique() > 1]

    # the ids are strings on some datasets and the matcher wants ints: number the
    # records, left and right on disjoint ranges
    left = pd.factorize(stored["ltable_id"])[0]
    right = pd.factorize(stored["rtable_id"])[0] + left.max() + 1
    return list(zip(left, right)), features, is_gold, len(gold)


def run_cell(dataset, side, k, configs):
    pairs, features, is_gold, n_gold = load_cell(dataset, side, k)
    pq, pc = is_gold.mean(), is_gold.sum() / n_gold
    cell = dict(dataset=dataset, side=side, k=k, pairs=len(pairs),
                n_features=features.shape[1], tp_blocking=int(is_gold.sum()),
                n_gold=n_gold, pc=pc, pq=pq,
                accept_all_f1=_f1(pq, pc), oracle_f1=_f1(1.0, pc))
    rows = []
    for name, config in configs.items():
        row = dict(cell, config=name, **config)
        matcher = ZeroERMatcher(**config)
        t0 = time.time()
        try:
            matcher.match_pairs(pairs, features)
        except RuntimeError as error:     # newton: no covariance update converged
            row.update(error=str(error))
        else:
            predicted = matcher.posteriors >= 0.5
            tp = int((predicted & is_gold).sum())
            p = tp / predicted.sum() if predicted.any() else 0.0
            r = tp / n_gold
            row.update(predicted=int(predicted.sum()), tp=tp,
                       precision=p, recall=r, f1=_f1(p, r),
                       init_frac=sum(matcher.model.y) / len(pairs),
                       pi_M=matcher.model.pi_M, n_iter=matcher.model.n_iter,
                       converged=matcher.model.converged)
        row.update(seconds=round(time.time() - t0, 1))
        rows.append(row)
        print("{dataset} {side}/{k} {config}: ".format(**row)
              + ("F1 {f1:.3f} (P {precision:.3f} R {recall:.3f}) accept-all "
                 "{accept_all_f1:.3f} iters {n_iter} converged {converged}".format(**row)
                 if "f1" in row else "CRASH " + row["error"]), flush=True)
    return rows


def main():
    if sys.argv[1:] == ["sensitivity"]:
        datasets, configs, name = SENSITIVITY_DATASETS, sensitivity_configs(), "ladder_sensitivity.csv"
    elif sys.argv[1:] == ["transitivity"]:
        configs = {"B-paper+transitivity": dict(CONFIGS["B-paper"], transitivity=True)}
        datasets, name = TRANSITIVITY_DATASETS, "ladder_transitivity.csv"
    else:
        datasets, configs, name = list(LADDER), CONFIGS, "ladder_baseline.csv"
    rows = [row for dataset in datasets
            for side, k in LADDER[dataset] for row in run_cell(dataset, side, k, configs)]
    out = join(OUTPUT_DIR, name)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
