"""Optuna sweep runners — generic over the objective.

Owns the study store (``optuna.db``) and the two sweep phases: tune on each
dataset's partition, then retrain the winning configs on the full data. The
objective itself is declared per experiment in the notebook, so a new
experiment never touches this file. Reading the results back out is
``reporting.py``.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import optuna

from data import DATASETS

# Storage — the single sqlite source of truth, shared by both (and soon other) notebooks
# pass ``storage=`` to point elsewhere.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
STORAGE = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna.db')}"


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


def _seed_everything(seed: int) -> None:
    """Seed every global RNG the pipeline touches (skill: seed first thing)."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Study runners — generic over the objective. The objective declares its own
# identity and search behaviour, so a new experiment never touches the runners:
#   directions   : optuna objective directions (F1 first, by convention)
#   experiment   : study-name suffix -> "<dataset>_<experiment>[_full]"
#   space        : search space dict (lets ``n_trials=None`` derive a budget)
#   seed_trials(): optional param dicts enqueued once on a fresh study,
#                  guaranteeing coverage TPE alone cannot (e.g. every model)
# --------------------------------------------------------------------------
def _study_name(dataset, objective, full=False) -> str:
    exp = getattr(objective, "experiment", "")
    name = f"{dataset}_{exp}" if exp else dataset
    return f"{name}_full" if full else name


def run_study(dataset, make_objective, n_trials, n_startup_trials,
              seed=0, storage=STORAGE) -> optuna.Study:
    """Tune one dataset on its ~tune_fraction partition; resumable in optuna.db.

    ``n_trials=None`` derives the budget from the objective's own ``space``
    (needed when the space is per-dataset, e.g. discovered embedding models).
    """
    objective = make_objective(dataset, partition=True)   # cheap: just loads the slice
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
    return study                          # persisted in optuna.db; CSVs via export()


def retrain_full(dataset, make_objective, seed=0,
                 storage=STORAGE) -> "optuna.Study | None":
    """Retrain a dataset's tuning Pareto front on the FULL data.

    The partition study only *ranks* configs on the ~30% slice; the reported
    end-to-end numbers must come from the full dataset. We take each
    Pareto-optimal config of ``<dataset>`` and re-evaluate it on the full data,
    recording the results in a sibling study ``<dataset>_full`` with the same
    objectives — so the full-data trade-off front lives in the same optuna.db.
    Configs are fixed, not searched: each is ``enqueue``-d and run through the
    *same* objective the notebook declared (so construction is identical), with
    the objective's ``source_trial`` set to stamp each full trial. That stamp
    lets the phase resume — already-retrained front configs are skipped.
    """
    objective = make_objective(dataset, partition=False)   # full data, loaded once
    tune_name = _study_name(dataset, objective)
    try:
        tune = optuna.load_study(study_name=tune_name, storage=storage)
    except KeyError:
        print(f"  skip {tune_name}_full: no tuning study yet (run the sweep first)",
              flush=True)
        return None
    front = front_trials(tune)
    full = optuna.create_study(
        study_name=_study_name(dataset, objective, full=True),
        storage=storage,
        load_if_exists=True,
        directions=objective.directions,
        # every trial's params are enqueued, so the sampler never actually
        # samples — a plain seeded TPE is enough.
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    done = {t.user_attrs.get("source_trial") for t in full.trials}
    todo = [t for t in front if t.number not in done]
    if not todo:
        print(f"  {full.study_name}: all {len(front)} front configs already retrained",
              flush=True)
        return full
    for src in todo:
        objective.source_trial = src.number
        full.enqueue_trial(src.params)
        # Same ``catch`` as tuning: a front config can still diverge on the full
        # data; a FAILED retrain of one config must not abort the rest.
        full.optimize(objective, n_trials=1, catch=(RuntimeError,))
    return full


def _blocking_stats(trial) -> Tuple[float, float]:
    """(candset_size, pair_completeness) of a trial, whichever experiment.

    New objectives record both as user attrs; the legacy 3-objective studies
    carried them as objective values 2 and 3 (candset_size was an attr too).
    """
    cs = trial.user_attrs.get(
        "candset_size", trial.values[1] if len(trial.values) > 1 else np.nan)
    pc = trial.user_attrs.get(
        "pair_completeness", trial.values[2] if len(trial.values) > 2 else np.nan)
    return cs, pc


# Single-objective winners: trials within this of the best F1 count as ties.
F1_TIE_DELTA = 0.005


def front_trials(study, delta: float = F1_TIE_DELTA) -> list:
    """The trials the views/retrain treat as a study's front.

    Multi-objective studies keep their Pareto front as-is. Single-objective
    (F1-max) studies get a blocking-aware tie-break: every completed trial
    within ``delta`` of the best F1 counts as a tie, and the one with the
    smallest candidate set wins — at equal F1, fewer surviving pairs is
    strictly preferable (same rationale as ``blocking_recall``: don't reward a
    blocker for keeping junk the matcher then has to reject).
    """
    if len(study.directions) > 1:
        return study.best_trials
    done = [t for t in study.trials if t.values is not None]
    if not done:
        return []
    top = max(t.values[0] for t in done)
    ties = [t for t in done if t.values[0] >= top - delta]
    return [min(ties, key=lambda t: (
        np.nan_to_num(_blocking_stats(t)[0], nan=np.inf), -t.values[0]))]


def _print_front(study, label: str) -> None:
    for t in front_trials(study):                    # Pareto front / tie-broken best
        cs, pc = _blocking_stats(t)
        print(f"  {label}: F1={t.values[0]:.4f} candset={int(cs)} "
              f"PC={pc:.4f} {t.params}", flush=True)


def tune_all(make_objective, n_trials, n_startup_trials, seed=0,
             storage=STORAGE) -> None:
    """Phase 1: tune every dataset on its ~30% partition (seeds first)."""
    _seed_everything(seed)
    for dataset in DATASETS:
        print(f"\n=== tuning {dataset} ({n_trials} trials) ===", flush=True)
        _print_front(
            run_study(dataset, make_objective, n_trials, n_startup_trials, seed, storage),
            "pareto")


def retrain_all(make_objective, seed=0, storage=STORAGE) -> None:
    """Phase 2: retrain each dataset's Pareto front on the full data (seeds first)."""
    _seed_everything(seed)
    for dataset in DATASETS:
        print(f"\n=== retraining {dataset} front on full data ===", flush=True)
        study = retrain_full(dataset, make_objective, seed, storage)
        if study is not None:
            _print_front(study, "full")
