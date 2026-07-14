"""
Utils for the ZeroER++ experiments
"""

from __future__ import annotations

import csv
import glob
import os
import random
import time
from typing import Tuple

import optuna

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pyjedai.datamodel import Data

from pyjedai_module import nn_search
from zeroer_features import build_zeroer_features

# Self-contained: the five gold-standard datasets live next to this file under
# zeroer-experiments/datasets/, resolved relative to this file (cwd-independent).
BENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "datasets"))

# Precomputed LLM record embeddings, one .npy per side per model, row-aligned
# with left.csv / right.csv (see datasets/embeddings/README.md):
#   datasets/embeddings/<dataset>/<model-stem>_{left,right}.npy
EMB_DIR = os.path.join(BENCH_DIR, "embeddings")

# Fraction of each benchmark kept for the Optuna *tuning* partition (see
# ``load_data(partition=True)``). The study tunes on this slice only; the best
# params are then retrained on the full dataset. Per-dataset overridable below.
DEFAULT_TUNE_FRACTION = 0.3

# Per-dataset metadata: matchable attributes, the tuning-partition fraction, plus
# any column rename needed to align the two tables on a shared name:
#   - fodors_zagats: 'type'/'class' dropped — 'class' is a ground-truth leak.
#   - amazon_googleproducts: Amazon's 'title' and Google's 'name' are the same
#     field under different names, so the right table is renamed 'name'->'title'.
DATASETS = {
    "fodors_zagats":         dict(attributes=["name", "addr", "city", "phone"],
                                  tune_fraction=0.3),
    "dblp_acm":              dict(attributes=["title", "authors", "venue", "year"],
                                  tune_fraction=0.3),
    "dblp_scholar":          dict(attributes=["title", "authors", "venue", "year"],
                                  tune_fraction=0.3),
    "abt_buy":               dict(attributes=["name", "description", "price"],
                                 tune_fraction=0.3),
    "amazon_googleproducts": dict(attributes=["title", "description", "manufacturer", "price"],
                                  tune_fraction=0.3, rename_2={"name": "title"})
}

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


def embedding_models(name: str, base: str = EMB_DIR) -> list:
    """Sorted model stems with precomputed embeddings for a dataset.

    A stem is the on-disk filename minus the ``_left.npy`` suffix (it starts
    with the embedding-model name, e.g. ``Qwen_Qwen3-Embedding-4B``); only
    models with both sides present count.
    """
    d = os.path.join(base, name)
    stems = sorted(os.path.basename(p)[:-len("_left.npy")]
                   for p in glob.glob(os.path.join(d, "*_left.npy")))
    return [s for s in stems
            if os.path.exists(os.path.join(d, f"{s}_right.npy"))]


def load_embeddings(name: str, model: str, data,
                    base: str = EMB_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Record vectors for ``data``'s two tables, as float32 (faiss-ready).

    The .npy files are row-aligned with the *full* left.csv / right.csv;
    ``data`` may be a subsampled partition, so rows are re-selected by matching
    the partition's id column against the full table's ids — an identity
    reindex on the full dataset.
    """
    d = os.path.join(base, name)
    out = []
    for side, table, id_col in (("left", data.dataset_1, data.id_column_name_1),
                                ("right", data.dataset_2, data.id_column_name_2)):
        vec = np.load(os.path.join(d, f"{model}_{side}.npy"), mmap_mode="r")
        full_ids = pd.read_csv(os.path.join(BENCH_DIR, name, f"{side}.csv"),
                               usecols=[id_col]).astype(str)[id_col]
        if len(vec) != len(full_ids):
            raise ValueError(
                f"{model}_{side}.npy has {len(vec)} rows but {name}/{side}.csv "
                f"has {len(full_ids)} — embeddings not row-aligned")
        pos = pd.Series(np.arange(len(full_ids)), index=full_ids)
        rows = pos[table[id_col].astype(str)].to_numpy()
        out.append(np.asarray(vec[rows], dtype=np.float32))
    return tuple(out)


class BlockFeatureCache:
    """Blocking + exact-feature reuse across a sweep's trials.

    k-NN candidate lists are nested in ``top_k`` (the top-3 neighbours are a
    prefix of the top-10 list), so blocking and ZeroER feature generation run
    **once** per (embedding model, similarity distance), at ``k_max``; every
    trial with ``top_k <= k_max`` is served as a row slice. Feature values are
    pair-local (no cross-pair statistics), so slices are exact — the constant-
    column drop, the one subset-dependent step, is re-applied per slice,
    matching ``build_zeroer_features(drop_zero_variance=True)`` on the same
    pairs. Builds are parallelised over pairs (``n_jobs``).

    Entries stay in memory for the cache's lifetime: at most
    (#models x #distances) matrices of ``k_max x n_right`` feature rows.
    """

    def __init__(self, dataset, data, attributes, k_max, n_jobs=-1):
        self.dataset = dataset
        self.data = data
        self.attributes = attributes
        self.k_max = k_max
        self.n_jobs = n_jobs
        self.limit = data.dataset_limit
        self._store = {}          # (model, distance) -> (nbrs, pairs, features)
        self.build_seconds = {}   # same key -> one-time blocking+feature cost

    def _built(self, model, distance):
        key = (model, distance)
        if key not in self._store:
            t0 = time.time()
            v1, v2 = load_embeddings(self.dataset, model, self.data)
            nbrs = nn_search(v1, v2, self.k_max, distance)
            pairs = np.column_stack([nbrs.ravel(),
                                     self.limit + np.repeat(np.arange(len(nbrs)),
                                                            nbrs.shape[1])])
            features = build_zeroer_features(
                pairs, self.data.entities, self.attributes,
                dataset_limit=self.limit, drop_zero_variance=False,
                n_jobs=self.n_jobs)
            self._store[key] = (nbrs, pairs, features)
            self.build_seconds[key] = round(time.time() - t0, 3)
        return self._store[key]

    def get(self, model, distance, top_k):
        """(pairs, features, blocks) for one trial, rows aligned; ``blocks``
        is the ``{right_global_id: neighbour set}`` dict the metrics expect."""
        nbrs, pairs, features = self._built(model, distance)
        if top_k > nbrs.shape[1]:
            raise ValueError(f"top_k={top_k} exceeds cached k_max={nbrs.shape[1]}")
        keep = np.tile(np.arange(nbrs.shape[1]), len(nbrs)) < top_k
        sliced = features.loc[keep]
        sliced = sliced.loc[:, sliced.nunique() > 1].reset_index(drop=True)
        blocks = {self.limit + j: set(row.tolist())
                  for j, row in enumerate(nbrs[:, :top_k])}
        return pairs[keep], sliced, blocks


def _subsample(
    left: pd.DataFrame,
    right: pd.DataFrame,
    gt: pd.DataFrame,
    fraction: float,
    seed: int,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair-preserving subsample of a clean-clean ER benchmark.

    Parameters
    ----------
    left, right
        Entity tables for the two sides of the clean-clean ER task. Each table
        is expected to contain a unique identifier column named ``id_col``.

    gt
        Ground-truth matches dataframe. The first two columns are assumed to be
        the left/right entity ids respectively (e.g. ``ltable_id``,
        ``rtable_id``).

    fraction
        Fraction of matched pairs and singleton entities to retain. Must lie in
        ``(0, 1]``.

    seed
        Random seed used for deterministic sampling.

    id_col
        Name of the entity-id column in both entity tables.

    Returns
    -------
    (left_s, right_s, gt_s)
        Subsampled left/right tables together with a recomputed ground-truth
        dataframe containing only surviving match pairs.

    Notes
    -----
    Idea here is to preserve the matched:unmatched ratio approximately (well, linearly)
    with the requested fraction, making tuning results transfer more generalisable
    to the full dataset.

    Steps:
      (1) sample ~``fraction`` of the gold pairs and keep both endpoints;
      (2) sample ~``fraction`` of unmatched singleton entities on each side;
      (3) recompute the ground truth from surviving endpoints.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")

    lcol, rcol = gt.columns[:2]

    # (1) ~fraction of matched pairs -> keep both endpoints
    gt_sampled = gt.sample(frac=fraction, random_state=seed)

    keep_l = (
        set(gt_sampled[lcol])
        | set(
            left.loc[~left[id_col].isin(gt[lcol]), id_col]
                .sample(frac=fraction, random_state=seed)
        )
    )

    keep_r = (
        set(gt_sampled[rcol])
        | set(
            right.loc[~right[id_col].isin(gt[rcol]), id_col]
                 .sample(frac=fraction, random_state=seed)
        )
    )

    # (2) assemble partition tables from surviving endpoints
    left_s = (
        left.loc[lambda df: df[id_col].isin(keep_l)]
            .reset_index(drop=True)
    )

    right_s = (
        right.loc[lambda df: df[id_col].isin(keep_r)]
             .reset_index(drop=True)
    )

    # (3) recompute GT consistent with surviving endpoints
    gt_s = (
        gt.loc[lambda df: df[lcol].isin(keep_l) & df[rcol].isin(keep_r)]
          .reset_index(drop=True)
    )

    return left_s, right_s, gt_s

def load_data(
    name: str,
    base: str = BENCH_DIR,
    partition: bool = False,
    seed: int = 0
) -> Data:
    """Load a zeroer-bench dataset into a pyJedAI clean-clean ``Data`` object.

    ``partition=True`` returns only the seeded ~``tune_fraction`` slice (see
    ``_subsample``) for hyperparameter tuning; the default returns the full
    dataset, which is the path for retraining the best params. The partition
    gets a distinct ``dataset_name`` so its PLM embeddings cache separately from
    the full-data embeddings (otherwise the full ``.npy`` would be indexed with
    the partition's smaller row indices, silently returning wrong vectors).
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    d = os.path.join(base, name)
    left = pd.read_csv(os.path.join(d, "left.csv")).astype(str)
    right = pd.read_csv(os.path.join(d, "right.csv")).astype(str)
    if spec.get("rename_1"):
        left = left.rename(columns=spec["rename_1"])
    if spec.get("rename_2"):
        right = right.rename(columns=spec["rename_2"])
    gt = pd.read_csv(os.path.join(d, "matches.csv")).astype(str)
    attrs = spec["attributes"]

    name_l, name_r = f"{name}:left", f"{name}:right"
    if partition:
        frac = spec.get("tune_fraction", DEFAULT_TUNE_FRACTION)
        left, right, gt = _subsample(left, right, gt, frac, seed)
        tag = f"p{int(round(frac * 100))}s{seed}"
        name_l, name_r = f"{name}:left:{tag}", f"{name}:right:{tag}"

    return Data(
        dataset_1=left, id_column_name_1="id", attributes_1=attrs,
        dataset_name_1=name_l,
        dataset_2=right, id_column_name_2="id", attributes_2=attrs,
        dataset_name_2=name_r,
        ground_truth=gt,
    )


def _count_gold_pairs_present(adj, data) -> int:
    """Count gold-standard pairs present in an adjacency-like container.

    Works for both a networkx ``Graph`` (the match graph) and a ``{id: set}``
    blocks dict — both answer ``i1 in adj`` and ``i2 in adj[i1]`` — so the two
    metrics below share exactly one membership test.
    """
    n = 0
    for _, (id1, id2) in data.ground_truth.iterrows():
        i1 = data._ids_mapping_1[id1]
        i2 = data._ids_mapping_2[id2]
        if (i1 in adj and i2 in adj[i1]) or (i2 in adj and i1 in adj[i2]):
            n += 1
    return n


def precision_recall_f1(graph, data) -> Tuple[float, float, float]:
    """End-to-end P/R/F1 of a match graph against the full gold standard.

    Recall denominator is the *complete* ground truth, so matches dropped by
    blocking count against recall (the statistically sound estimand when the
    blocker is itself a variable being compared).
    """
    tp = _count_gold_pairs_present(graph, data)
    pred = graph.number_of_edges()
    truth = len(data.ground_truth)
    p = tp / pred if pred else 0.0
    r = tp / truth if truth else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def blocking_recall(blocks, data) -> float:
    """Pair completeness: the fraction of gold matches that survive blocking.

    Computed on the candidate-pair structure *before* the matcher runs, so it
    isolates the matches the blocker discards (unrecoverable) from those the
    matcher later rejects. It is therefore the ceiling that end-to-end recall
    can reach for a given blocking configuration — the "don't drop true pairs to
    begin with" objective of the sweep.
    """
    if not blocks:
        return 0.0
    truth = len(data.ground_truth)
    return _count_gold_pairs_present(blocks, data) / truth if truth else 0.0


def _seed_everything(seed: int) -> None:
    """Seed every global RNG the pipeline touches (skill: seed first thing)."""
    random.seed(seed)
    np.random.seed(seed)


def _round(x, n):
    """round() that tolerates None (conditional params absent from a branch)."""
    return round(x, n) if x is not None else None


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


def _pareto_row(dataset: str, trial, phase: str = "partition") -> dict:
    """Flatten one Pareto-optimal trial into a CSV row.

    Generic over experiments: F1 is the first objective value by convention;
    the trial's params and user attrs (candset_size, pair_completeness,
    timings, source_trial, ...) go in as-is, so a new experiment's knobs need
    no view changes.
    """
    cs, pc = _blocking_stats(trial)
    row = dict(
        dataset=dataset,
        phase=phase,                       # "partition" (tuned) or "full" (retrained)
        F1=round(trial.values[0], 4),
        candset_size=int(cs),
        blocking_recall=_round(pc, 4),
    )
    row.update({k: (_round(v, 5) if isinstance(v, float) else v)
                for k, v in trial.params.items()})
    row.update({k: v for k, v in trial.user_attrs.items()
                if k not in ("candset_size", "pair_completeness")})
    return row


def export(storage=STORAGE, experiment: str = "") -> None:
    """Regenerate the CSV views from optuna.db (the source of truth).

    Reads whatever studies are already in the DB — run it after a full or even a
    partial sweep. For each dataset it dumps the partition study
    (``<dataset>[_<experiment>]``) and, if present, its full-data retrain
    (``..._full``) as verbatim ``trials_dataframe`` logs, then collects the
    Pareto-optimal trials of each phase into ``pareto_fronts[_<experiment>].csv``
    (partition) and ``full_pareto_fronts[_<experiment>].csv`` (full) — the
    latter being the configs' honest end-to-end numbers.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"_{experiment}" if experiment else ""
    partition_rows, full_rows, n_studies = [], [], 0
    for dataset in DATASETS:
        for phase, suffix, sink in (("partition", "", partition_rows),
                                    ("full", "_full", full_rows)):
            study_name = f"{dataset}{tag}{suffix}"
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
            except KeyError:
                continue                             # phase not run yet — skip
            n_studies += 1
            study.trials_dataframe().to_csv(
                os.path.join(OUTPUT_DIR, f"{study_name}_trials.csv"), index=False)
            sink.extend(_pareto_row(dataset, t, phase) for t in front_trials(study))

    if not partition_rows and not full_rows:
        print(f"No studies found in {storage} — run the sweep first.")
        return
    written = []
    for rows, fname in ((partition_rows, f"pareto_fronts{tag}.csv"),
                        (full_rows, f"full_pareto_fronts{tag}.csv")):
        if not rows:
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", newline="") as fh:
            # rows may differ in keys across studies — union, first-seen order
            fieldnames = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(rows)
        written.append(f"{fname} ({len(rows)} Pareto trials)")
    print(f"Wrote {', '.join(written)} and {n_studies} *_trials.csv from {storage}")


# --------------------------------------------------------------------------
# Results views — read straight from optuna.db, so they work after a full *or*
# partial sweep and are identical in both notebooks.
#
# Optional:
#     show_results()                  -> all studies in the default STORAGE
#     show_results(study_name="abt_buy")  -> only abt_buy / abt_buy_full
# --------------------------------------------------------------------------

def _load_studies(
    suffix: str,
    storage=STORAGE,
    study_name: str | None = None,
    experiment: str = "",
):
    """(dataset, study) for every matching study in optuna.db that has at least
    one completed trial."""
    out = []

    datasets = [study_name] if study_name is not None else DATASETS
    tag = f"_{experiment}" if experiment else ""

    for dataset in datasets:
        try:
            s = optuna.load_study(
                study_name=f"{dataset}{tag}{suffix}",
                storage=storage,
            )
        except KeyError:
            continue

        if any(t.values is not None for t in s.trials):
            out.append((dataset, s))

    return out


def summarize_fronts(
    phase: str,
    suffix: str,
    storage=STORAGE,
    study_name: str | None = None,
    experiment: str = "",
) -> None:
    studies = _load_studies(suffix, storage, study_name, experiment)

    rows = [
        _pareto_row(dataset, t, phase)
        for dataset, s in studies
        for t in front_trials(s)
    ]

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "F1"],
        ascending=[True, False],
    )

    print(
        f"=== {phase}: {len(df)} Pareto-optimal configs across "
        f"{len(studies)} dataset(s) ==="
    )

    display(df.reset_index(drop=True))


def plot_pareto_fronts(
    phase: str,
    suffix: str,
    storage=STORAGE,
    study_name: str | None = None,
    experiment: str = "",
) -> None:
    studies = _load_studies(suffix, storage, study_name, experiment)

    if not studies:
        return

    fig, axes = plt.subplots(
        len(studies),
        2,
        figsize=(11, 3.3 * len(studies)),
        squeeze=False,
    )

    for r, (dataset, s) in enumerate(studies):
        done = [t for t in s.trials if t.values is not None]
        front = {t.number for t in front_trials(s)}

        f1 = np.array([t.values[0] for t in done])
        stats = [_blocking_stats(t) for t in done]
        cs = np.array([c for c, _ in stats], dtype=float)
        pc = np.array([p for _, p in stats], dtype=float)
        opt = np.array([t.number in front for t in done])

        for c, (x, xlabel, logx) in enumerate(
            (
                (cs, "surviving block pairs (candset)", True),
                (pc, "blocking recall (pair completeness)", False),
            )
        ):
            ax = axes[r][c]

            ax.scatter(x[~opt], f1[~opt], s=18, c="lightgray", label="trial")

            ax.scatter(
                x[opt],
                f1[opt],
                s=46,
                c="tab:red",
                zorder=3,
                edgecolor="k",
                linewidth=0.4,
                label="Pareto front",
            )

            ax.set(
                xlabel=xlabel,
                ylabel="F1",
                title=f"{dataset} — {phase}",
            )

            if logx:
                ax.set_xscale("log")

            ax.grid(alpha=0.3)

        axes[r][0].legend(loc="lower right", fontsize=8)

    label = f"{experiment} {phase}" if experiment else phase
    title = (
        f"ZeroER++ {label} Pareto fronts"
        if study_name is None
        else f"ZeroER++ {label} Pareto fronts — {study_name}"
    )

    fig.suptitle(title, y=1.0, fontsize=13)
    fig.tight_layout()
    plt.show()


def show_results(
    storage=STORAGE,
    study_name: str | None = None,
    experiment: str = "",
) -> None:
    """Best configs + Pareto-front plots for whichever phases exist."""

    shown = False

    for phase, suffix in (
        ("partition", ""),
        ("full", "_full"),
    ):
        if _load_studies(suffix, storage, study_name, experiment):
            summarize_fronts(phase, suffix, storage, study_name, experiment)
            plot_pareto_fronts(phase, suffix, storage, study_name, experiment)
            shown = True

    if not shown:
        target = f"study '{study_name}'" if study_name else "studies"
        print(f"No matching {target} in storage yet.")
