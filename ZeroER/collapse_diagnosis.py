"""Why string-only ZeroER collapses on the product data — look before repairing.

    data-env/bin/python collapse_diagnosis.py
        -> outputs/collapse_diagnosis/{blocks,presence_confusion,treatments}.csv
        -> studies "<dataset>_collapse" in outputs/optuna_matching.db

Step 2 of NEXT-STEPS-matching.md, first half. ``ladder_baseline.py`` shows the
reference at or below the accept-all floor on every ``abt_buy`` and
``amazon_googleproducts`` rung, under the paper's configuration as much as the
released code's. The plan's candidate explanation is *inferred, not observed*:
Σ is block-diagonal by attribute, so every block adds its log-likelihood-ratio
at full weight, and a block whose cleanest two-cluster split is "value present
vs absent" (absent values are exact-zero feature rows) drags the posterior
there. Two things here, on the collapsed rungs plus two working controls:

**A — inspect the fitted model**, per attribute block (``module.py`` groups
columns by ``name.split("_")[0]``): its parameters, its share of the
log-likelihood-ratio — over all pairs, gold matches / non-matches, and pairs
where the attribute is present / absent — and the confusion of the fitted
components against gold *and* against attribute presence, with the same table
for the initial labels, so a split the EM drifts into can be told from one the
initialisation hands it. The zero-fill
reference is the fit the hypothesis is about; the treated fits of part B are read
the same way, so that a change in F1 can be traced to what the components became.
``kappa_at_cap`` is the share of a block's columns whose regulariser sits at
``m_step``'s upper clip of 1 — a variance of 1 added to a feature that lives in
[0, 1], i.e. a block flattened out of the likelihood ratio. ``run_em`` ends on an
M-step, so the posteriors it returns are one step behind the parameters it
leaves; the diagnostics run one more E-step and read everything off that, so
parameters, log-ratios and labels all belong to the same state.

**B — the mean-imputation proxy** the plan designed and never ran: an absent
block's zeros are replaced by the block's column means over the present rows,
beside the key-attribute-only matrix. An improvement supports *investigating*
missingness; it does not establish the mechanism, because imputing also changes
the initialisation (no more zero-sum rows) and the covariances.

Presence comes from ``reference_features._tables`` — the tables the stored
matrices were built from (``abt_buy``'s right ``description`` is upstream's
``description + manufacturer`` concat, which propagates NaN) — and is checked
against the matrix rather than assumed: see ``absent_rows_all_zero``.

Gold scores and diagnoses; it never chooses anything. Fits are fixed
configurations written with ``add_trial`` and skipped when already stored, as in
``ladder_baseline.py``; F1 is against the *full* gold standard.
"""
import os
import time
from os.path import exists, join

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState
from scipy.stats import multivariate_normal

from ladder_baseline import CONFIGS, DISTRIBUTIONS, _f1, config_hash, load_side
from matcher import ZeroERMatcher
from reference_features import BENCH_DIR, LADDER, _tables
from tuning import MATCHING_STORAGE, OUTPUT_DIR

EXPERIMENT = "collapse"        # -> study "<dataset>_collapse"
OUT_DIR = join(OUTPUT_DIR, "collapse_diagnosis")

# every collapsed rung, and two cells where the reference works, as controls
CELLS = {
    "abt_buy": LADDER["abt_buy"],
    "amazon_googleproducts": LADDER["amazon_googleproducts"],
    "dblp_acm": [("left", 2)],
    "dblp_scholar": [("left", 5)],
}
KEY_ATTRIBUTE = {"abt_buy": "name"}     # 'title' everywhere else
TREATMENTS = ["zero_fill", "mean_impute", "key_attribute", "key_attribute_mean_impute"]
TRIAL_DISTRIBUTIONS = dict(DISTRIBUTIONS, treatment=CategoricalDistribution(TREATMENTS))


def _blocks(columns):
    """{attribute: its column names}, grouped the way ``module.py`` groups them."""
    blocks = {}
    for name in columns:
        blocks.setdefault(name.split("_")[0], []).append(name)
    return blocks


def presence(dataset, ids, attributes):
    """Per pair and attribute: is the value there on *both* sides?"""
    left, right = _tables(dataset, BENCH_DIR)
    left, right = (t.set_index(t["id"].astype(str)) for t in (left, right))
    return pd.DataFrame({
        attr: left[attr].reindex(ids["ltable_id"]).notna().to_numpy()
        & right[attr].reindex(ids["rtable_id"]).notna().to_numpy()
        for attr in attributes})


def treat(features, present, treatment, key):
    """The feature matrix one treatment fits, constant columns dropped."""
    features = features.astype(float)
    if treatment.startswith("key_attribute"):
        features = features[_blocks(features.columns)[key]]
    if treatment.endswith("mean_impute"):
        features = features.copy()
        for attr, columns in _blocks(features.columns).items():
            absent = ~present[attr].to_numpy()
            if absent.any() and not absent.all():
                features.loc[absent, columns] = features.loc[~absent, columns].mean().to_numpy()
    return features.loc[:, features.nunique() > 1]


def _phi(a, b):
    """2x2 counts of two boolean arrays and their phi coefficient."""
    n11, n10 = int((a & b).sum()), int((a & ~b).sum())
    n01, n00 = int((~a & b).sum()), int((~a & ~b).sum())
    denominator = np.sqrt(float(n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return dict(n11=n11, n10=n10, n01=n01, n00=n00,
                phi=(n11 * n00 - n10 * n01) / denominator if denominator else np.nan)


def diagnose(matcher, features, is_gold, present):
    """(per-block rows, confusion rows) for one fitted model."""
    model = matcher.model
    model.e_step()      # refresh: Q_M / Q_U / P_M now follow the final parameters
    predicted = model.P_M >= 0.5
    total = model.Q_M - model.Q_U
    blocks, summed = [], np.zeros(len(total))
    for attr, names in _blocks(features.columns).items():
        cols = [features.columns.get_loc(name) for name in names]
        x, here = model.X[:, cols], present[attr].to_numpy()
        llr = (multivariate_normal.logpdf(x, model.Mu_M[cols], model.Cov_M[np.ix_(cols, cols)],
                                          allow_singular=True)
               - multivariate_normal.logpdf(x, model.Mu_U[cols], model.Cov_U[np.ix_(cols, cols)],
                                            allow_singular=True))
        summed += llr
        zero_row = ~x.any(axis=1)
        blocks.append(dict(
            block=attr, n_columns=len(cols), present_share=here.mean(),
            zero_row_share=zero_row.mean(),
            absent_rows_all_zero=zero_row[~here].mean() if (~here).any() else np.nan,
            present_rows_all_zero=zero_row[here].mean() if here.any() else np.nan,
            mu_M=model.Mu_M[cols].mean(), mu_U=model.Mu_U[cols].mean(),
            std_M=np.sqrt(np.diag(model.Cov_M)[cols]).mean(),
            std_U=np.sqrt(np.diag(model.Cov_U)[cols]).mean(),
            kappa=model.kappas[cols].mean(), kappa_at_cap=(model.kappas[cols] >= 1).mean(),
            llr_all=llr.mean(), llr_gold_match=llr[is_gold].mean(),
            llr_gold_nonmatch=llr[~is_gold].mean(),
            llr_present=llr[here].mean() if here.any() else np.nan,
            llr_absent=llr[~here].mean() if (~here).any() else np.nan))
    cell = dict(pi_M=float(model.pi_M), predicted=int(predicted.sum()),
                labels_changed_by_refresh=int((predicted != (matcher.posteriors >= 0.5)).sum()),
                llr_blocks_vs_total_max_abs=float(np.abs(summed - total).max()))
    # the initial labels too: is the split already there before the first M-step?
    against = dict({"gold": is_gold}, **{attr: present[attr].to_numpy() for attr in present},
                   all_attributes=present.all(axis=1).to_numpy())
    confusion = [dict(cell, labels=name, against=what, **_phi(labels, truth))
                 for name, labels in (("initial", np.asarray(model.y, dtype=bool)),
                                      ("fitted", predicted))
                 for what, truth in against.items()]
    return blocks, confusion


def _append(path, rows):
    frame = pd.DataFrame(rows)
    if exists(path):
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _diagnosed(path):
    if not exists(path):
        return set()
    done = pd.read_csv(path)
    return set(zip(done.dataset, done.side, done.k, done.config, done.treatment))


def run(storage=MATCHING_STORAGE):
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks_csv, confusion_csv = join(OUT_DIR, "blocks.csv"), join(OUT_DIR, "presence_confusion.csv")
    # a generous lock timeout: the ladder run writes to the same file
    storage = optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 120}})
    for dataset, cells in CELLS.items():
        study = optuna.create_study(study_name="{}_{}".format(dataset, EXPERIMENT),
                                    storage=storage, load_if_exists=True,
                                    direction="maximize")
        done = {config_hash(t.params) for t in study.trials}
        diagnosed = _diagnosed(blocks_csv)
        key = KEY_ATTRIBUTE.get(dataset, "title")
        for side in dict.fromkeys(s for s, _ in cells):
            s = load_side(dataset, side)
            present_side = presence(dataset, s["ids"], _blocks(s["features"].columns))
            for k in [k for c, k in cells if c == side]:
                keep = s["rank"] < k
                left, right, is_gold = s["left"][keep], s["right"][keep], s["is_gold"][keep]
                present = present_side[keep].reset_index(drop=True)
                cell = dict(candset_size=int(keep.sum()), tp_blocking=int(is_gold.sum()),
                            n_gold=s["n_gold"])
                for name, config in CONFIGS.items():
                    for treatment in TREATMENTS:
                        params = dict(queried_side=side, top_k=k, treatment=treatment, **config)
                        look = (dataset, side, k, name, treatment) not in diagnosed
                        if config_hash(params) in done and not look:
                            continue
                        features = treat(s["features"][keep].reset_index(drop=True),
                                         present, treatment, key)
                        attrs = dict(cell, config=name, config_hash=config_hash(params),
                                     n_features=features.shape[1])
                        matcher, t0 = ZeroERMatcher(**config), time.time()
                        try:
                            matcher.match_pairs(list(zip(left, right)), features)
                        except (RuntimeError, ValueError) as error:   # newton / singular fit
                            attrs.update(error=repr(error), seconds=round(time.time() - t0, 1))
                            f1 = None
                        else:
                            predicted, model = matcher.posteriors >= 0.5, matcher.model
                            attrs.update(
                                predicted=int(predicted.sum()), tp=int((predicted & is_gold).sum()),
                                init_frac=float(np.mean(model.y)), pi_M=float(model.pi_M),
                                n_iter=int(model.n_iter), converged=bool(model.converged),
                                seconds=round(time.time() - t0, 1))
                            f1 = _f1(attrs["tp"], attrs["predicted"], s["n_gold"])
                        if config_hash(params) not in done:
                            study.add_trial(optuna.trial.create_trial(
                                params=params, distributions=TRIAL_DISTRIBUTIONS,
                                user_attrs=attrs, value=f1,
                                state=TrialState.FAIL if f1 is None else TrialState.COMPLETE))
                        if look and f1 is not None:
                            where = dict(dataset=dataset, side=side, k=k, config=name,
                                         treatment=treatment)
                            blocks, confusion = diagnose(matcher, features, is_gold, present)
                            _append(blocks_csv, [dict(where, **row) for row in blocks])
                            _append(confusion_csv, [dict(where, **row) for row in confusion])
                        print("{} {}/{} {} {}: {}".format(
                            dataset, side, k, name, treatment,
                            "CRASH " + attrs["error"] if f1 is None else
                            "F1 {:.3f}  accept-all {:.3f}".format(
                                f1, _f1(cell["tp_blocking"], cell["candset_size"], s["n_gold"]))),
                            flush=True)
    treatments_frame(storage).to_csv(join(OUT_DIR, "treatments.csv"), index=False)


def treatments_frame(storage):
    """One row per stored fit, the metrics recomputed from the raw counts."""
    rows = []
    for dataset in CELLS:
        study = optuna.load_study(study_name="{}_{}".format(dataset, EXPERIMENT), storage=storage)
        rows += [dict(dataset=dataset, **t.params, **t.user_attrs) for t in study.trials]
    df = pd.DataFrame(rows)
    df["PQ"] = df.tp_blocking / df.candset_size
    df["PC"] = df.tp_blocking / df.n_gold
    df["precision"] = df.tp / df.predicted
    df["recall"] = df.tp / df.n_gold
    df["f1"] = 2 * df.tp / (df.predicted + df.n_gold)
    df["accept_all_f1"] = 2 * df.tp_blocking / (df.candset_size + df.n_gold)
    df["oracle_f1"] = 2 * df.PC / (1 + df.PC)
    return df


if __name__ == "__main__":
    run()
