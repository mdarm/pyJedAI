"""Optuna searches, fixed sweeps and full-data replays.

Experiment choices live in the notebooks; fitting and scoring use evaluation.py.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from os.path import exists, join

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

from config import LADDER, MATCHING_STORAGE, OUTPUT_DIR, STORAGE, cell_name
from data import PARAMETERS_M, embedding_models, load_data, load_embeddings
from diagnostics import _blocks, diagnose, treat
from embedding_features import FUSIONS
from estimator import nn_search
from evaluation import (_f1, config_hash, evaluate_config, fit, load_side, load_sides,
                        one_to_one, one_to_one_oracle, presence)
from matcher import ZeroERMatcher
from metrics import gold_ranks


def _matching_params(trial, space):
    """Sample the matcher parameters described by the notebook's search space."""
    params = dict(
        c_bay=trial.suggest_float("c_bay", *space["c_bay"], log=True),
        init_threshold=trial.suggest_float("init_threshold", *space["init_threshold"]),
        init_statistic=trial.suggest_categorical("init_statistic", space["init_statistic"]),
        pi_M=trial.suggest_categorical("pi_M", space["pi_M"]),
        hard=trial.suggest_categorical("hard", space["hard"]))
    for key in ("missing", "transitivity", "max_iter", "tol"):
        if key in space:
            params[key] = trial.suggest_categorical(key, space[key])
    return params


def _evaluate_trial(dataset, loaded, params, fixed, missing=None, attributes=None,
                    holdout=None, include_strings=True):
    """Reconstruct a sampled or stored configuration and use the same evaluator."""
    config = {key: value for key, value in params.items()
              if key != "fusion" and not key.startswith("use_")}
    config.update(fixed)
    if missing is not None:
        config["missing"] = missing
    models = [key[4:] for key in sorted(params) if key.startswith("use_") and params[key]]
    panel = dict(models=models, fusion=params["fusion"]) if models else None
    value, attrs = evaluate_config(dataset, loaded, config, attributes=attributes,
                                   panel=panel, holdout=holdout, include_strings=include_strings)
    attrs["config_hash"] = config_hash(params)
    if "fusion" in params:
        attrs.update(n_models=len(models), parameters_m=sum(PARAMETERS_M[m] for m in models),
                     include_strings=include_strings, attributes=attributes)
    else:
        attrs["kappa_capped"] = float(np.mean([
            row.get("kappa_capped", np.nan) for row in attrs["rungs"]]))
    return value, attrs


class MatchingObjective:
    """Tune a string-feature matcher on half of each direction's queried records."""

    directions = ["maximize"]

    def __init__(self, dataset, experiment, space, fixed, missing="zero", seed=0,
                 n_shared=0, seeds=()):
        self.dataset, self.experiment, self.space = dataset, experiment, space
        self.fixed, self.missing, self.seed = fixed, missing, seed
        self.n_shared, self.seeds = n_shared, list(seeds)
        self.loaded = load_sides(dataset, partition="tune", seed=seed)

    def seed_trials(self):
        """Named controls followed by a shared, seeded random design."""
        rng = np.random.default_rng(self.seed)
        lo, hi = np.log(self.space["c_bay"])
        return self.seeds + [dict(
            c_bay=float(np.exp(rng.uniform(lo, hi))),
            init_threshold=float(rng.uniform(*self.space["init_threshold"])),
            init_statistic=str(rng.choice(self.space["init_statistic"])),
            pi_M=self.space["pi_M"][rng.integers(len(self.space["pi_M"]))],
            hard=bool(rng.integers(2))) for _ in range(self.n_shared - len(self.seeds))]

    def __call__(self, trial):
        params = _matching_params(trial, self.space)
        value, attrs = _evaluate_trial(self.dataset, self.loaded, params, self.fixed,
                                       missing=self.missing)
        for key, item in attrs.items():
            trial.set_user_attr(key, item)
        return value


class EmbeddingObjective:
    """Evaluate the notebook's panel design or search over its encoder pool."""

    directions = ["maximize"]

    def __init__(self, dataset, experiment, space, fixed, models, attributes, design, seed=0,
                 include_strings=True):
        self.dataset, self.experiment, self.space = dataset, experiment, space
        self.fixed, self.models, self.attributes = fixed, list(models), attributes
        self.design = design
        self.include_strings = include_strings
        self.loaded = load_sides(dataset, partition="tune", seed=seed, bank=True)

    def seed_trials(self):
        return list(self.design.values())

    def __call__(self, trial):
        params = _matching_params(trial, self.space)
        params["fusion"] = trial.suggest_categorical("fusion", self.space.get("fusion", list(FUSIONS)))
        params.update({"use_" + model: trial.suggest_categorical("use_" + model, [False, True])
                       for model in self.models})
        if not self.include_strings and not any(params["use_" + model] for model in self.models):
            raise optuna.TrialPruned("Cosine-only matching requires at least one model")
        value, attrs = _evaluate_trial(self.dataset, self.loaded, params, self.fixed,
                                       attributes=self.attributes, include_strings=self.include_strings)
        if trial.number < len(self.design):
            attrs["rule"] = list(self.design)[trial.number]
        for key, item in attrs.items():
            trial.set_user_attr(key, item)
        return value


def replay_selected(dataset, experiment, selections, fixed, missing=None,
                    attributes=None, seed=0, storage=MATCHING_STORAGE):
    """Refit {selection: source trial number} on full and complementary-half data.

    Parameters, distributions and embedding input choices come from the tuning
    study. No search space or panel design is reconstructed. Stored selections
    are skipped on a rerun.
    """
    tune = optuna.load_study(study_name=f"{dataset}_{experiment}", storage=storage)
    sources = {trial.number: trial for trial in tune.trials}
    full = optuna.create_study(study_name=f"{dataset}_{experiment}_full", storage=storage,
                               load_if_exists=True, direction="maximize")
    done = {trial.user_attrs.get("selection") for trial in full.trials}
    loaded = holdout = None
    for selection, number in selections.items():
        if selection in done:
            continue
        source = sources[number]
        if source.state != TrialState.COMPLETE:
            raise ValueError(f"Cannot replay incomplete trial {number} of {tune.study_name}")
        if loaded is None:
            bank = "fusion" in source.params
            loaded = load_sides(dataset, seed=seed, bank=bank)
            holdout = load_sides(dataset, partition="holdout", seed=seed, bank=bank)
        value, attrs = _evaluate_trial(dataset, loaded, source.params, fixed, missing=missing,
                                       attributes=source.user_attrs.get("attributes", attributes),
                                       holdout=holdout,
                                       include_strings=source.user_attrs.get("include_strings", True))
        attrs.update(source_trial=source.number, selection=selection)
        full.add_trial(optuna.trial.create_trial(
            params=source.params, distributions=source.distributions, value=value,
            user_attrs=attrs))
    return full


def trial_budget(
    space: dict,
    n_startup_trials: int,
    coverage: float = 1 / 3
) -> int:
    """TPE warm-up + ~``coverage`` of the discrete grid (a budget derived from
    the space, not a magic number). Generic over experiments: categorical lists
    and int ``(lo, hi)`` ranges multiply into the grid; float ranges are
    *continuous* dims on top, so sampling does real work there and ``coverage``
    just sets how much of the discrete cells to visit.
    """
    cells = 1
    for spec in space.values():
        if isinstance(spec, list):
            cells *= len(spec)
        elif isinstance(spec, tuple) and all(isinstance(v, int) for v in spec):
            lo, hi = spec
            cells *= hi - lo + 1
    return n_startup_trials + int(cells * coverage)


def _study_name(dataset, objective) -> str:
    exp = getattr(objective, "experiment", "")
    name = f"{dataset}_{exp}" if exp else dataset
    return name


def run_study(dataset, make_objective, n_trials, n_startup_trials,
              seed=0, storage=STORAGE) -> optuna.Study:
    """Tune one dataset on its ~tune_fraction partition; resumable in optuna.db.

    ``n_trials=None`` derives the budget from the objective's own ``space``
    (needed when the space is per-dataset, e.g. discovered embedding models).
    """
    objective = make_objective(dataset)
    if n_trials is None:
        n_trials = trial_budget(objective.space, n_startup_trials)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,           # model param interactions jointly
        group=True,                  # handle the conditional (per-branch) space
    )
    study = optuna.create_study(
        study_name=_study_name(dataset, objective),
        storage=storage,
        load_if_exists=True,
        directions=objective.directions,
        sampler=sampler,
    )
    if not study.trials:             # fresh study only: resuming re-enqueues nothing
        for params in getattr(objective, "seed_trials", list)():
            study.enqueue_trial(params)
    # A degenerate config can make e.g. ZeroER's EM diverge (scipy.optimize.newton
    # raises RuntimeError when *all* covariance updates fail to converge) or
    # yield an all-constant feature matrix (RuntimeError).
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    return study


# Stored coordinate schema for fixed baseline and diagnosis fits.
DISTRIBUTIONS = {
    "queried_side": CategoricalDistribution(["right", "left"]),
    "top_k": IntDistribution(1, 100),
    "c_bay": FloatDistribution(0.0, 1.0),
    "init_threshold": FloatDistribution(0.0, 1.0),
    "tol": CategoricalDistribution([None, 1e-5]),
    "max_iter": IntDistribution(1, 1000),
    "transitivity": CategoricalDistribution([False, True]),
}


def run_baselines(configs, datasets=None, rungs_only=False, storage=MATCHING_STORAGE,
                  posteriors_dir=join(OUTPUT_DIR, "ladder_posteriors")):
    """Fit every config on every cell, skipping what the study already holds."""
    os.makedirs(posteriors_dir, exist_ok=True)
    # a generous lock timeout: other experiments write to the same file
    storage = optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 120}})
    for dataset in datasets or LADDER:
        study = optuna.create_study(study_name="{}_{}".format(dataset, "ladder"),
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
                _run_baseline_side(study, dataset, side, todo, posteriors_dir)


def _run_baseline_side(study, dataset, side, todo, posteriors_dir):
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
                out = join(posteriors_dir, "{}__{}.parquet".format(
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


def run_diagnosis(configs, cells, treatments, key_attributes, storage=MATCHING_STORAGE,
                  out_dir=join(OUTPUT_DIR, "collapse_diagnosis")):
    """Fit the notebook's treatments and save counts plus component diagnostics."""
    distributions = dict(DISTRIBUTIONS, treatment=CategoricalDistribution(treatments))
    os.makedirs(out_dir, exist_ok=True)
    blocks_csv, confusion_csv = join(out_dir, "blocks.csv"), join(out_dir, "presence_confusion.csv")
    # a generous lock timeout: the ladder run writes to the same file
    storage = optuna.storages.RDBStorage(
        storage, engine_kwargs={"connect_args": {"timeout": 120}})
    for dataset, dataset_cells in cells.items():
        study = optuna.create_study(study_name="{}_{}".format(dataset, "collapse"),
                                    storage=storage, load_if_exists=True,
                                    direction="maximize")
        done = {config_hash(t.params) for t in study.trials}
        diagnosed = _diagnosed(blocks_csv)
        key = key_attributes.get(dataset, "title")
        for side in dict.fromkeys(s for s, _ in dataset_cells):
            s = load_side(dataset, side)
            present_side = presence(dataset, s["ids"], _blocks(s["features"].columns))
            for k in [k for c, k in dataset_cells if c == side]:
                keep = s["rank"] < k
                left, right, is_gold = s["left"][keep], s["right"][keep], s["is_gold"][keep]
                present = present_side[keep].reset_index(drop=True)
                cell = dict(candset_size=int(keep.sum()), tp_blocking=int(is_gold.sum()),
                            n_gold=s["n_gold"])
                for name, config in configs.items():
                    for treatment in treatments:
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
                                params=params, distributions=distributions,
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



def run_blocking_sweep(dataset, k_max, queried_sides,
                       storage=STORAGE):
    'Store the complete k sweep; skip completed model/direction pairs.'
    data = load_data(dataset, partition=False)
    models = embedding_models(dataset)
    limit, n_gold = data.dataset_limit, len(data.ground_truth)
    n_side = {"right": data.num_of_entities_2, "left": data.num_of_entities_1}

    study = optuna.create_study(
        study_name=f"{dataset}_blocking",
        storage=storage, load_if_exists=True, directions=["maximize", "minimize"])
    counts = Counter((t.params["embedding_model"], t.params["queried_side"])
                     for t in study.trials)
    partial = {cell: n for cell, n in counts.items() if n != k_max}
    if partial:
        raise RuntimeError(
            f"{study.study_name}: {len(partial)} cell(s) do not hold exactly "
            f"{k_max} trials, e.g. {dict(list(partial.items())[:3])} — either a "
            "run was interrupted mid-write, or the study was built with a "
            "different k_max. Resuming would leave a truncated or double-counted "
            "k-sweep, so delete the study (optuna.delete_study) and re-run.")
    done = set(counts)
    dists = dict(
        embedding_model=CategoricalDistribution(models),
        top_k=IntDistribution(1, k_max),
        queried_side=CategoricalDistribution(queried_sides),
    )

    print(f"=== {dataset}: {data.num_of_entities_1}+{data.num_of_entities_2} "
          f"entities, {n_gold} matches, {len(models)} models "
          f"({len(done)}/{len(models) * len(queried_sides)} cells done) ===",
          flush=True)

    for model in models:
        vectors = load_seconds = None
        for d in queried_sides:
            if (model, d) in done:
                continue
            if vectors is None:
                t0 = time.time()
                vectors = load_embeddings(dataset, model, data)
                load_seconds = round(time.time() - t0, 3)
            v1, v2 = vectors
            index_side, query_side = (v1, v2) if d == "right" else (v2, v1)

            t0 = time.time()
            nbrs = nn_search(index_side, query_side, k_max, "cosine")
            faiss_seconds = round(time.time() - t0, 3)

            srt = np.sort(nbrs, axis=1)
            assert not (srt[:, 1:] == srt[:, :-1]).any(), "duplicate neighbours"

            tp_at_k = np.cumsum(np.bincount(
                gold_ranks(nbrs, data, d, limit),
                minlength=nbrs.shape[1] + 2)[1:nbrs.shape[1] + 1])

            trials = []
            for k, tp in enumerate(tp_at_k, start=1):
                tp, candset = int(tp), int(n_side[d]) * k
                trials.append(optuna.trial.create_trial(
                    params=dict(embedding_model=model, top_k=k, queried_side=d),
                    distributions=dists,
                    values=[tp / n_gold, float(candset)],
                    user_attrs=dict(
                        tp=tp, n_gold=n_gold, candset_size=candset,
                        pair_completeness=round(tp / n_gold, 6),
                        pair_quality=round(tp / candset, 6),
                        load_seconds=load_seconds, faiss_seconds=faiss_seconds,
                        dim=int(v1.shape[1]),
                        n_left=int(data.num_of_entities_1),
                        n_right=int(data.num_of_entities_2),
                    )))
            study.add_trials(trials)
            print(f"  {model:40} {d:5} PC@1={tp_at_k[0] / n_gold:.4f} "
                  f"PC@{len(tp_at_k)}={tp_at_k[-1] / n_gold:.4f} "
                  f"load={load_seconds}s faiss={faiss_seconds}s", flush=True)
            del nbrs, srt
        del vectors
    return study
