"""String-only ZeroER on the ladder's candidate sets — the reference every later
matching experiment is read against. One study per dataset,
``<dataset>_ladder``, in ``outputs/optuna_matching.db``.

    data-env/bin/python ladder_baseline.py              # full nested-k curves
    data-env/bin/python ladder_baseline.py sensitivity  # κ′ / ε around B-paper
    data-env/bin/python ladder_baseline.py transitivity # duplicate-free transitivity

``B-paper`` is the paper's stated configuration (1908.06049v2 sections 3.3, 5.1):
κ′ = 0.01, initialisation threshold ε = 0.5, stop when the expected
log-likelihood moves < 1e-5, at most 200 iterations. ``B-code`` is what the
released code runs: ``c_bay`` = 0.1, threshold 0.8, its own free-energy stop, 40
iterations. As in the paper, ZeroER is fitted on the whole candidate set — its
50/50 train/test split is for the supervised baselines only.

This is a *sweep*, not a search: every fit is a fixed configuration written with
``add_trial``, so no sampler is consulted, and a re-run skips what is stored.

**No k-NN and no feature generation.** A cell is its stored pairs and genuine
Magellan matrix (``reference_features.py``). Pairs are stored grouped by queried
record in neighbour-rank order and k-NN lists are nested, so the matrix at the
largest ``k`` of a (dataset, side) holds every smaller ``k`` as a row slice —
the whole curve comes from the 19 stored rungs.

**What is stored.** Raw counts, so every metric is derivable afterwards
(``reporting.ladder_frame``): ``tp`` / ``predicted`` for ZeroER against the
*full* gold standard (``n_gold`` — matches the blocker missed count against
recall), ``tp_blocking`` / ``candset_size`` for the cell itself — which give the
accept-all floor and the oracle ceiling — and ``floor_*`` for the second
no-matcher floor: the blocker's whole-record cosine with a greedy one-to-one
mapping, no threshold. The objective value is ZeroER's F1. Each fit's posterior
over *every* candidate pair goes to ``outputs/ladder_posteriors/`` (its path is
the ``posteriors`` attribute); a fit that raises is stored as a FAIL trial.
"""
import hashlib
import json
import os
import sys
import time
from os.path import join

import networkx as nx
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)
from optuna.trial import TrialState

from data import load_data, load_embeddings
from matcher import ZeroERMatcher
from reference_features import BENCH_DIR, FEATURES_DIR, LADDER, MODEL, cell_name
from tuning import MATCHING_STORAGE, OUTPUT_DIR

EXPERIMENT = "ladder"          # -> study "<dataset>_ladder"
POSTERIORS_DIR = join(OUTPUT_DIR, "ladder_posteriors")

CONFIGS = {
    "B-paper": dict(c_bay=0.01, init_threshold=0.5, tol=1e-5, max_iter=200,
                    transitivity=False),
    "B-code": dict(c_bay=0.1, init_threshold=0.8, tol=None, max_iter=40,
                   transitivity=False),
}

# A trial's params are the cell's coordinates — named as the blocking sweep names
# them — plus the ZeroERMatcher arguments, so a later tuning study that reuses
# these names can have any of its configs replayed here.
DISTRIBUTIONS = {
    "queried_side": CategoricalDistribution(["right", "left"]),
    "top_k": IntDistribution(1, 100),
    "c_bay": FloatDistribution(0.0, 1.0),
    "init_threshold": FloatDistribution(0.0, 1.0),
    "tol": CategoricalDistribution([None, 1e-5]),
    "max_iter": IntDistribution(1, 1000),
    "transitivity": CategoricalDistribution([False, True]),
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
TRANSITIVITY_CONFIGS = {"B-paper+transitivity": dict(CONFIGS["B-paper"], transitivity=True)}

# Column groups in the stored matrices that are outside data.DATASETS' attribute
# allow-list: fodors_zagats' 'class' is a ground-truth leak, 'type' goes with it.
DROP_GROUPS = {"fodors_zagats": ["type", "class"]}


def _f1(tp, predicted, n_gold):
    return 2 * tp / (predicted + n_gold) if predicted + n_gold else 0.0


def config_hash(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def blocking_cosine(dataset, stored):
    """The blocker's whole-record cosine of every stored pair."""
    data = load_data(dataset)
    v1, v2 = load_embeddings(dataset, MODEL, data)
    rows_l = pd.Series(np.arange(len(v1)), index=data.dataset_1[data.id_column_name_1])
    rows_r = pd.Series(np.arange(len(v2)), index=data.dataset_2[data.id_column_name_2])
    u = v1[rows_l[stored["ltable_id"]].to_numpy()]
    v = v2[rows_r[stored["rtable_id"]].to_numpy()]
    return np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))


def load_side(dataset, side):
    """Everything the cells of one (dataset, queried side) share, read off the
    stored matrix at that side's largest ``k``."""
    k_max = max(k for s, k in LADDER[dataset] if s == side)
    stored = pd.read_csv(join(FEATURES_DIR, cell_name(dataset, side, k_max) + ".csv.gz"),
                         dtype={"ltable_id": str, "rtable_id": str})
    gt = pd.read_csv(join(BENCH_DIR, dataset, "matches.csv"), dtype=str)
    gold = set(zip(gt.iloc[:, 0], gt.iloc[:, 1]))

    features = stored.drop(columns=["ltable_id", "rtable_id"])
    group = features.columns.str.split("_").str[0]
    features = features.loc[:, ~group.isin(DROP_GROUPS.get(dataset, []))]

    cos = blocking_cosine(dataset, stored)
    # each queried record's pairs must come nearest-first, or the rank slices
    # below are not the smaller-k candidate sets — and a re-encoded record/ tree
    # under unchanged model names would show up exactly here
    if (np.diff(cos.reshape(-1, k_max), axis=1) > 1e-4).any():
        raise ValueError(
            "{} {}/{}: stored pairs are not in the blocker's neighbour order — "
            "have the record vectors changed since the pairs were exported?"
            .format(dataset, side, k_max))

    # ids are strings on some datasets and the matcher wants ints: number the
    # records, left and right on disjoint ranges
    left = pd.factorize(stored["ltable_id"])[0]
    right = pd.factorize(stored["rtable_id"])[0] + left.max() + 1
    return dict(k_max=k_max, ids=stored[["ltable_id", "rtable_id"]], left=left,
                right=right, features=features, blocking_cos=cos, n_gold=len(gold),
                rank=np.arange(len(stored)) % k_max,
                is_gold=np.array([pair in gold for pair in
                                  zip(stored["ltable_id"], stored["rtable_id"])]))


def one_to_one(left, right, score):
    """Greedy unique mapping: best-scoring pair first, every record used once."""
    keep = np.zeros(len(score), dtype=bool)
    used = set()
    for i in np.argsort(-score, kind="stable"):
        if left[i] not in used and right[i] not in used:
            keep[i] = True
            used.update((left[i], right[i]))
    return keep


def one_to_one_oracle(left, right, is_gold):
    """The most gold pairs any one-to-one mapping of these candidates can keep —
    below ``is_gold.sum()`` wherever the gold standard is not one-to-one."""
    graph = nx.Graph(zip(left[is_gold], right[is_gold]))
    return len(nx.bipartite.hopcroft_karp_matching(graph, set(left[is_gold]))) // 2


def fit(config, pairs, features):
    """One EM fit -> (posteriors, fit attributes); raises what the EM raises."""
    matcher = ZeroERMatcher(**config)
    matcher.match_pairs(pairs, features)
    model = matcher.model
    return matcher.posteriors, dict(
        init_frac=float(np.mean(model.y)), pi_M=float(model.pi_M),
        n_iter=int(model.n_iter), converged=bool(model.converged))


def run(configs=CONFIGS, datasets=None, rungs_only=False, storage=MATCHING_STORAGE):
    """Fit every config on every cell, skipping what the study already holds."""
    os.makedirs(POSTERIORS_DIR, exist_ok=True)
    # a generous lock timeout: other experiments write to the same file
    storage = optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 120}})
    for dataset in datasets or LADDER:
        study = optuna.create_study(study_name="{}_{}".format(dataset, EXPERIMENT),
                                    storage=storage, load_if_exists=True,
                                    direction="maximize")
        done = {config_hash(t.params) for t in study.trials}
        for side in dict.fromkeys(s for s, _ in LADDER[dataset]):
            k_max = max(k for s, k in LADDER[dataset] if s == side)
            cells = [k for k in range(1, k_max + 1)
                     if (side, k) in LADDER[dataset] or not rungs_only]
            todo = [(k, name, dict(queried_side=side, top_k=k, **config))
                    for k in cells for name, config in configs.items()]
            todo = [t for t in todo if config_hash(t[2]) not in done]
            if todo:
                _run_side(study, dataset, side, todo)


def _run_side(study, dataset, side, todo):
    s = load_side(dataset, side)
    for k in dict.fromkeys(k for k, _, _ in todo):
        keep = s["rank"] < k
        left, right, is_gold = s["left"][keep], s["right"][keep], s["is_gold"][keep]
        features = s["features"][keep]
        features = features.loc[:, features.nunique() > 1]
        floor = one_to_one(left, right, s["blocking_cos"][keep])
        cell = dict(candset_size=int(keep.sum()), tp_blocking=int(is_gold.sum()),
                    n_gold=s["n_gold"], n_features=features.shape[1],
                    rung=(side, k) in LADDER[dataset],
                    floor_predicted=int(floor.sum()),
                    floor_tp=int((floor & is_gold).sum()),
                    floor_oracle_tp=one_to_one_oracle(left, right, is_gold))
        for _, name, params in (t for t in todo if t[0] == k):
            config = {a: params[a] for a in params if a not in ("queried_side", "top_k")}
            attrs = dict(cell, config=name, config_hash=config_hash(params))
            t0 = time.time()
            try:
                posteriors, fitted = fit(config, list(zip(left, right)), features)
            except RuntimeError as error:   # newton: no covariance update converged
                attrs.update(error=str(error), seconds=round(time.time() - t0, 1))
                f1 = None
            else:
                predicted = posteriors >= 0.5
                out = join(POSTERIORS_DIR, "{}__{}.parquet".format(
                    cell_name(dataset, side, k), attrs["config_hash"]))
                s["ids"][keep].assign(blocking_cos=s["blocking_cos"][keep],
                                      p_match=posteriors).to_parquet(out, index=False)
                attrs.update(fitted, predicted=int(predicted.sum()),
                             tp=int((predicted & is_gold).sum()),
                             seconds=round(time.time() - t0, 1),
                             posteriors=os.path.relpath(out, OUTPUT_DIR))
                f1 = _f1(attrs["tp"], attrs["predicted"], s["n_gold"])
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=DISTRIBUTIONS, user_attrs=attrs, value=f1,
                state=TrialState.FAIL if f1 is None else TrialState.COMPLETE))
            print("{} {}/{} {}: {}".format(
                dataset, side, k, name,
                "CRASH " + attrs["error"] if f1 is None else
                "F1 {:.3f}  accept-all {:.3f}  cosine 1:1 {:.3f}  iters {}{}".format(
                    f1, _f1(cell["tp_blocking"], cell["candset_size"], s["n_gold"]),
                    _f1(cell["floor_tp"], cell["floor_predicted"], s["n_gold"]),
                    attrs["n_iter"], "" if attrs["converged"] else " (cap)")),
                flush=True)


def main():
    if sys.argv[1:] == ["sensitivity"]:
        run(sensitivity_configs(), SENSITIVITY_DATASETS, rungs_only=True)
    elif sys.argv[1:] == ["transitivity"]:
        run(TRANSITIVITY_CONFIGS, TRANSITIVITY_DATASETS, rungs_only=True)
    else:
        run()


if __name__ == "__main__":
    main()
