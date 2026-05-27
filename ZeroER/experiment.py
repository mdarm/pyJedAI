"""ZeroER++ multi-objective hyperparameter study + full-data retraining.

Two phases, run back-to-back by default (each is also selectable on the CLI):

  1. TUNE  — one Optuna study per dataset, on the seeded ~30% partition. The
     search space and budget are the module-level constants below;
     ``TuningObjective.get_params`` is the single source of truth for *what* is
     searched. Each study trades match quality, blocking cost, and blocking
     completeness as a Pareto problem:

         objective 1 (maximize): end-to-end F1
         objective 2 (minimize): surviving block pairs (candset size)
         objective 3 (maximize): blocking recall / pair completeness — the
             fraction of gold matches that survive blocking (the ceiling on
             recall, i.e. how few true pairs the blocker drops "to begin with")

     so every study returns a Pareto front over (quality, cost, completeness)
     rather than a single "best".

  2. RETRAIN-FULL — the partition only *ranks* configs cheaply; the honest
     end-to-end numbers come from the full data. So each tuning study's Pareto
     front is re-evaluated on the full dataset and stored in a sibling study
     ``<dataset>_full`` (same three objectives). Every full trial carries a
     ``source_trial`` user-attr linking back to the partition trial it came
     from, which also makes the phase resumable.

CLI:
    python experiment.py                 # both phases (tune, then retrain-full)
    python experiment.py --tune-only     # phase 1 only
    python experiment.py --retrain-full  # phase 2 only (needs existing studies)
    python experiment.py --export        # rebuild CSV views from optuna.db

Persistence — the SQLite DB is the single source of truth:
  * all studies (partition + ``_full``) share one database (``outputs/optuna.db``),
    so a run can be resumed and inspected later (e.g. ``optuna-dashboard``);
  * the CSV views are *not* written during the run. Regenerate them from the DB
    on demand (even after a partial sweep) with ``python experiment.py --export``
    — every per-dataset trial log and both Pareto fronts are derived from the DB.

Run from the pyJedAI repo root with the ``pyjedai`` env active. Note phase 1
sweeps all five datasets across nine PLMs (on the ~30% partition), so a cold run
is long — that is why it is kept ready-to-run rather than run inline.
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import optuna

from pyjedai_module import ZeroEREstimator
from utils import (DATASETS, DEFAULT_TUNE_FRACTION, blocking_recall, load_data,
                   precision_recall_f1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
STORAGE = f"sqlite:///{os.path.join(OUTPUT_DIR, 'optuna.db')}"

# --------------------------------------------------------------------------
# Search space — every entry maps one-to-one onto a flat ZeroEREstimator arg.
# --------------------------------------------------------------------------
# All transformer PLMs pyJedAI's EmbeddingsNNBlockBuilding exposes
# (vector_based_blocking.py). The gensim/static vectorizers (word2vec, glove,
# fasttext, doc2vec, sent_glove) are intentionally excluded: they are static
# word embeddings, not pre-trained *language models*.
PLM_VECTORIZERS = [
    # sentence-transformer PLMs
    "sminilm", "smpnet", "st5", "sdistilroberta",
    # word/token transformer PLMs
    "bert", "distilbert", "roberta", "xlnet", "albert",
]
TOP_K_MIN, TOP_K_MAX = 1, 10            # neighbours kept per entity (int)
# FAISS is the only NN backend pyJedAI exposes, so there is nothing to tune for
# the search engine itself; only its metric is a knob.
SIMILARITY_DISTANCES = ["cosine", "euclidean"]   # NN metric (categorical)
C_BAY_MIN, C_BAY_MAX = 1e-3, 3e-1       # ZeroER EM regularisation (log-float)

# Standard (syntactic) branch — block-cleaning + comparison-cleaning knobs.
SMOOTHING_MIN, SMOOTHING_MAX = 1.0, 1.5      # BlockPurging purge threshold (float)
RATIO_MIN, RATIO_MAX = 0.5, 0.95             # BlockFiltering keep-ratio (float)
WEIGHTING_SCHEMES = ["CBS", "ECBS", "JS", "EJS", "X2"]   # WeightedEdgePruning

# --------------------------------------------------------------------------
# Budget — derived from the space, not a magic number (cf. the skill's
# "default by deriving" rule).
# --------------------------------------------------------------------------
SEED = 0
# TPE draws this many random trials before it starts modelling the space.
N_STARTUP_TRIALS = 12
# Distinct discrete cells across both blocker branches ...
_EMB_CELLS = len(PLM_VECTORIZERS) * (TOP_K_MAX - TOP_K_MIN + 1) * len(SIMILARITY_DISTANCES)  # 9*10*2 = 180
_STD_CELLS = len(WEIGHTING_SCHEMES)        # 5  (smoothing_factor & ratio are continuous)
_DISCRETE_CELLS = _EMB_CELLS + _STD_CELLS  # 185
# ... and c_bay / smoothing_factor / ratio add *continuous* dimensions on top,
# so the space cannot be enumerated — sampling is doing real work. Budget =
# TPE's random warm-up plus ~a third of the discrete grid to explore the
# continuous dims within promising cells. Raise COVERAGE for a finer search,
# lower it for a quick smoke run.
COVERAGE = 1 / 3
N_TRIALS = N_STARTUP_TRIALS + int(_DISCRETE_CELLS * COVERAGE)   # 12 + 61 = 73


def _seed_everything(seed: int) -> None:
    """Seed every global RNG the pipeline touches (skill: seed first thing)."""
    random.seed(seed)
    np.random.seed(seed)


def _round(x, n):
    """round() that tolerates None (conditional params absent from a branch)."""
    return round(x, n) if x is not None else None


class TuningObjective:
    """Multi-objective trial: (max F1, min surviving block pairs, max pair completeness).

    One instance per dataset; the ``Data`` is loaded once and reused across all
    trials. ``get_params`` *is* the search space — it maps one-to-one onto the
    flat ``ZeroEREstimator`` constructor.
    """

    def __init__(self, dataset: str, partition: bool = True):
        self.dataset = dataset
        # partition=True tunes on the seeded ~tune_fraction slice (cheaper + the
        # statistically sound surface); partition=False is the full-data retrain
        # of an already-chosen config. ``source_trial`` is set per-call by the
        # retrain phase to stamp each full trial with its partition origin.
        self.partition = partition
        self.source_trial = None
        self.data = load_data(dataset, partition=partition, seed=SEED)
        self.attributes = DATASETS[dataset]["attributes"]
        if partition:
            frac = DATASETS[dataset].get("tune_fraction", DEFAULT_TUNE_FRACTION)
            scope = f"partition (frac={frac})"
        else:
            scope = "full data"
        print(f"    {scope}: "
              f"{self.data.num_of_entities_1}+{self.data.num_of_entities_2} entities, "
              f"{len(self.data.ground_truth)} matches", flush=True)

    def get_params(self, trial) -> dict:
        # THE ENTIRE SEARCH SPACE LIVES HERE — one readable block. It is
        # *conditional*: the blocker is itself a knob, and each branch exposes
        # only its own stage parameters (semantic PLM blocking vs syntactic
        # blocking + cleaning). c_bay (the matcher) is searched either way.
        blocker = trial.suggest_categorical("blocker", ["embeddings", "standard"])
        params = dict(
            blocker=blocker,
            c_bay=trial.suggest_float("c_bay", C_BAY_MIN, C_BAY_MAX, log=True),
        )
        if blocker == "embeddings":
            params.update(
                vectorizer=trial.suggest_categorical("vectorizer", PLM_VECTORIZERS),
                top_k=trial.suggest_int("top_k", TOP_K_MIN, TOP_K_MAX),
                similarity_distance=trial.suggest_categorical(
                    "similarity_distance", SIMILARITY_DISTANCES),
            )
        else:                                   # syntactic blocking + cleaning
            params.update(
                smoothing_factor=trial.suggest_float(
                    "smoothing_factor", SMOOTHING_MIN, SMOOTHING_MAX),
                block_filtering_ratio=trial.suggest_float(
                    "block_filtering_ratio", RATIO_MIN, RATIO_MAX),
                weighting_scheme=trial.suggest_categorical(
                    "weighting_scheme", WEIGHTING_SCHEMES),
            )
        return params

    def __call__(self, trial):
        est = ZeroEREstimator(attributes=self.attributes, **self.get_params(trial))
        graph = est.fit_predict(self.data)
        _, _, f1 = precision_recall_f1(graph, self.data)
        # Pair completeness on the candidate pairs *before* matching (est.blocks),
        # i.e. the share of gold matches the blocker did not throw away.
        pair_completeness = blocking_recall(est.blocks, self.data)
        # Capture extras now so the trial log carries them alongside the params.
        trial.set_user_attr("candset_size", est.candset_size)
        trial.set_user_attr("blocking_seconds", round(est.blocking_seconds, 3))
        trial.set_user_attr("em_seconds", round(est.matcher.em_time, 3))
        if self.source_trial is not None:            # full-retrain provenance
            trial.set_user_attr("source_trial", self.source_trial)
        return f1, est.candset_size, pair_completeness


def run_study(dataset: str) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=True,           # model param interactions jointly
        group=True,                  # handle the conditional (per-branch) space
    )
    study = optuna.create_study(
        study_name=dataset,
        storage=STORAGE,
        load_if_exists=True,
        # F1 up, block pairs down, blocking recall (pair completeness) up
        directions=["maximize", "minimize", "maximize"],
        sampler=sampler,
    )
    study.optimize(TuningObjective(dataset), n_trials=N_TRIALS)
    return study                          # persisted in optuna.db; CSVs via --export


def retrain_full(dataset: str) -> "optuna.Study | None":
    """Retrain a dataset's tuning Pareto front on the FULL data.

    The partition study only *ranks* configs on the ~30% slice; the reported
    end-to-end numbers must come from the full dataset. We take each
    Pareto-optimal config of ``<dataset>`` and re-evaluate it on the full data,
    recording the results in a sibling study ``<dataset>_full`` with the same
    three objectives — so the full-data trade-off front lives in the same
    optuna.db. Configs are fixed, not searched: each is ``enqueue``-d and run
    through the *same* objective path as tuning (so estimator construction is
    identical), and stamped with a ``source_trial`` user-attr. That stamp lets
    the phase resume — already-retrained front configs are skipped.
    """
    try:
        tune = optuna.load_study(study_name=dataset, storage=STORAGE)
    except KeyError:
        print(f"  skip {dataset}_full: no tuning study yet (run the sweep first)",
              flush=True)
        return None
    front = tune.best_trials
    full = optuna.create_study(
        study_name=f"{dataset}_full",
        storage=STORAGE,
        load_if_exists=True,
        directions=["maximize", "minimize", "maximize"],
        # every trial's params are enqueued, so the sampler never actually
        # samples — a plain seeded TPE is enough.
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    done = {t.user_attrs.get("source_trial") for t in full.trials}
    todo = [t for t in front if t.number not in done]
    if not todo:
        print(f"  {dataset}_full: all {len(front)} front configs already retrained",
              flush=True)
        return full
    objective = TuningObjective(dataset, partition=False)   # full data, loaded once
    for src in todo:
        objective.source_trial = src.number
        full.enqueue_trial(src.params)
        full.optimize(objective, n_trials=1)
    return full


def _pareto_row(dataset: str, trial, phase: str = "partition") -> dict:
    """Flatten one Pareto-optimal trial into a CSV row.

    Conditional params are read with ``.get`` (a branch's params are absent from
    the other branch's trials); ``_round`` tolerates the resulting ``None``s.
    """
    f1, candset, pair_completeness = trial.values
    p = trial.params
    return dict(
        dataset=dataset,
        phase=phase,                       # "partition" (tuned) or "full" (retrained)
        F1=round(f1, 4),
        candset_size=int(candset),
        blocking_recall=round(pair_completeness, 4),
        blocker=p["blocker"],
        # embeddings branch
        vectorizer=p.get("vectorizer"),
        top_k=p.get("top_k"),
        similarity_distance=p.get("similarity_distance"),
        # standard branch
        smoothing_factor=_round(p.get("smoothing_factor"), 4),
        block_filtering_ratio=_round(p.get("block_filtering_ratio"), 4),
        weighting_scheme=p.get("weighting_scheme"),
        # matcher
        c_bay=_round(p.get("c_bay"), 5),
        blocking_seconds=trial.user_attrs.get("blocking_seconds"),
        em_seconds=trial.user_attrs.get("em_seconds"),
        # full trials only: the partition trial this config was chosen from
        source_trial=trial.user_attrs.get("source_trial"),
    )


def export() -> None:
    """Regenerate the CSV views from optuna.db (the source of truth).

    Reads whatever studies are already in the DB — run it after a full or even a
    partial sweep. For each dataset it dumps the partition study (``<dataset>``)
    and, if present, its full-data retrain (``<dataset>_full``) as verbatim
    ``trials_dataframe`` logs, then collects the Pareto-optimal trials of each
    phase into ``pareto_fronts.csv`` (partition) and ``full_pareto_fronts.csv``
    (full) — the latter being the configs' honest end-to-end numbers.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    partition_rows, full_rows, n_studies = [], [], 0
    for dataset in DATASETS:
        for phase, suffix, sink in (("partition", "", partition_rows),
                                    ("full", "_full", full_rows)):
            study_name = f"{dataset}{suffix}"
            try:
                study = optuna.load_study(study_name=study_name, storage=STORAGE)
            except KeyError:
                continue                             # phase not run yet — skip
            n_studies += 1
            study.trials_dataframe().to_csv(
                os.path.join(OUTPUT_DIR, f"{study_name}_trials.csv"), index=False)
            sink.extend(_pareto_row(dataset, t, phase) for t in study.best_trials)

    if not partition_rows and not full_rows:
        print(f"No studies found in {STORAGE} — run the sweep first.")
        return
    written = []
    for rows, fname in ((partition_rows, "pareto_fronts.csv"),
                        (full_rows, "full_pareto_fronts.csv")):
        if not rows:
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        written.append(f"{fname} ({len(rows)} Pareto trials)")
    print(f"Wrote {', '.join(written)} and {n_studies} *_trials.csv from {STORAGE}")


def _print_front(study, label: str) -> None:
    for t in study.best_trials:                      # the Pareto-optimal trials
        f1, candset, pair_completeness = t.values
        print(f"  {label}: F1={f1:.4f} candset={int(candset)} "
              f"PC={pair_completeness:.4f} {t.params}", flush=True)


def tune_all() -> None:
    """Phase 1: tune every dataset on its ~30% partition."""
    for dataset in DATASETS:
        print(f"\n=== tuning {dataset} ({N_TRIALS} trials) ===", flush=True)
        _print_front(run_study(dataset), "pareto")


def retrain_all() -> None:
    """Phase 2: retrain each dataset's Pareto front on the full data."""
    for dataset in DATASETS:
        print(f"\n=== retraining {dataset} front on full data ===", flush=True)
        study = retrain_full(dataset)
        if study is not None:
            _print_front(study, "full")


def main(tune: bool = True, retrain: bool = True) -> None:
    _seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if tune:
        tune_all()
    if retrain:
        retrain_all()
    print(f"\nDone. optuna.db (source of truth): {STORAGE}")
    print("Regenerate CSV views with:  python experiment.py --export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroER++ Optuna study driver.")
    # Add a separate arguments module at some point.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tune-only", action="store_true",
        help="phase 1 only: tune on the partition; do not retrain on full data")
    mode.add_argument(
        "--retrain-full", action="store_true",
        help="phase 2 only: retrain each existing Pareto front on the full data")
    mode.add_argument(
        "--export", action="store_true",
        help="regenerate the CSV views from optuna.db and exit (no tuning)")
    args = parser.parse_args()
    if args.export:
        export()
    else:
        main(tune=not args.retrain_full, retrain=not args.tune_only)
