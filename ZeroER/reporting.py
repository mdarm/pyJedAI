"""Results views — read straight from optuna.db, so they work after a full *or*
partial sweep and are identical in both notebooks.

``export`` regenerates the CSV views; ``show_results`` is the in-notebook
summary + Pareto-front plots.

Optional:
    show_results()                      -> all studies in the default STORAGE
    show_results(study_name="abt_buy")  -> only abt_buy / abt_buy_full
"""

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from data import DATASETS
from tuning import OUTPUT_DIR, STORAGE, _blocking_stats, front_trials


def _round(x, n):
    """round() that tolerates None (conditional params absent from a branch)."""
    return round(x, n) if x is not None else None


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


# --------------------------------------------------------------------------
# Blocking-only sweep views
#
# The blocking sweep records raw counts, not ratios, so everything below is
# derived here — a new metric never means re-running the grid. It needs its own
# views because ``_pareto_row`` hardcodes ``values[0]`` as an F1 column, which
# would silently mislabel pair completeness.
# --------------------------------------------------------------------------

BLOCKING_EXPERIMENT = "blocking"
PC_TARGET = 0.95            # "target recall" for the k* / candset@k* columns


def blocking_frame(storage=STORAGE, experiment=BLOCKING_EXPERIMENT,
                   datasets=None) -> pd.DataFrame:
    """One tidy row per swept cell, with the per-cell metrics recomputed.

    PC (pair completeness / recall), PQ (pair quality / precision) and RR
    (reduction ratio) all follow from the stored ``tp`` / ``candset_size`` /
    ``n_gold`` counts.
    """
    rows = []
    for dataset in (datasets or DATASETS):
        try:
            study = optuna.load_study(study_name=f"{dataset}_{experiment}",
                                      storage=storage)
        except KeyError:
            continue                              # dataset not swept yet
        rows += [dict(dataset=dataset, **t.params, **t.user_attrs)
                 for t in study.trials if t.values is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["PC"] = df.tp / df.n_gold
    df["PQ"] = df.tp / df.candset_size
    df["RR"] = 1 - df.candset_size / (df.n_left * df.n_right)
    return df.sort_values(["dataset", "embedding_model", "queried_side", "top_k"])


def _average_precision(pc, pq) -> float:
    """Average precision of one model's k-sweep: ``sum_k (PC_k - PC_{k-1})*PQ_k``.

    This is the area under the PC-vs-PQ curve — UniBlocker's mAP — in the
    standard IR form that anchors the curve at PC_0 = 0. The anchoring matters
    here: integrating only between the swept endpoints would score a blocker
    that already reaches PC=1 at k=1 as *zero*, since its PC never moves.

    Bounded above by ``n_gold / n_queried`` (all matches retrieved at k=1), so
    it is comparable across models within a (dataset, queried side) but not across
    datasets — see ``nAP`` in ``blocking_summary`` for the normalised form.
    """
    return float((np.diff(pc, prepend=0.0) * pq).sum())


def blocking_summary(df: pd.DataFrame, pc_target: float = PC_TARGET) -> pd.DataFrame:
    """Collapse the k axis: one row per (dataset, model, queried side).

    ``AP``/``nAP`` rank models over the whole sweep; ``PC@k`` reproduces the
    fixed-k table format; ``k*``/``candset@k*`` answer the "smallest candidate
    set reaching a target recall" question, which is the protocol that actually
    compares blocking cost at equal quality.
    """
    out = []
    for (dataset, model, side), g in df.groupby(
            ["dataset", "embedding_model", "queried_side"], sort=False):
        g = g.sort_values("top_k")
        pc, pq = g.PC.to_numpy(), g.PQ.to_numpy()
        n_queried = int(g.n_right.iloc[0] if side == "right" else g.n_left.iloc[0])
        ap = _average_precision(pc, pq)
        hit = g[g.PC >= pc_target]
        row = dict(
            dataset=dataset, model=model, queried_side=side,
            AP=round(ap, 4),
            # best possible AP is n_gold/n_queried (everything found at k=1),
            # which differs per dataset — normalise to compare across them
            nAP=round(ap * n_queried / int(g.n_gold.iloc[0]), 4),
            PC_max=round(float(pc[-1]), 4),
        )
        for k in (1, 5, 10):
            at_k = g[g.top_k == k]
            row[f"PC@{k}"] = round(float(at_k.PC.iloc[0]), 4) if len(at_k) else np.nan
        row["k*"] = int(hit.top_k.iloc[0]) if len(hit) else np.nan
        row["candset@k*"] = int(hit.candset_size.iloc[0]) if len(hit) else np.nan
        row["faiss_s"] = float(g.faiss_seconds.iloc[0])
        row["load_s"] = float(g.load_seconds.iloc[0])
        row["dim"] = int(g.dim.iloc[0])
        out.append(row)
    return pd.DataFrame(out)


def blocking_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank the embedding models across all datasets.

    Each model is taken at its better queried side per dataset, then aggregated
    three ways: **mean rank** (assumption-free — the per-dataset ordering is all
    that transfers), **wins** (datasets topped, ties included — on an easy dataset many models reach PC=1 at k=1 and genuinely tie), and **mean nAP** (the scale-free
    effectiveness average). Rank and nAP disagreeing is the signal that one
    model is winning on a subset rather than uniformly.
    """
    best = (summary.sort_values("AP", ascending=False)
                   .drop_duplicates(["dataset", "model"]))
    best = best.assign(rank=best.groupby("dataset").AP.rank(ascending=False,
                                                            method="min"))
    board = best.groupby("model").agg(
        mean_rank=("rank", "mean"),
        wins=("rank", lambda r: int((r == 1).sum())),
        mean_nAP=("nAP", "mean"),
        mean_PC1=("PC@1", "mean"),
        datasets=("dataset", "nunique"),
        faiss_s=("faiss_s", "mean"),
        dim=("dim", "first"),
    ).sort_values(["mean_rank", "mean_nAP"], ascending=[True, False])
    return board.round(4).reset_index()


def plot_blocking(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Two views per the blocking literature: the PC-vs-cost trade-off curve
    (one panel per dataset, every model's k-sweep, envelope highlighted) and the
    effectiveness-vs-time scatter, where the best models sit lower-right."""
    datasets = list(dict.fromkeys(df.dataset))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 3.4 * len(datasets)),
                             squeeze=False)
    for r, dataset in enumerate(datasets):
        d = df[df.dataset == dataset]
        ax = axes[r][0]
        for (_, _), g in d.groupby(["embedding_model", "queried_side"], sort=False):
            g = g.sort_values("top_k")
            ax.plot(g.candset_size, g.PC, c="lightgray", lw=0.8, zorder=1)
        env = d.sort_values("PC", ascending=False).drop_duplicates("candset_size")
        env = env.sort_values("candset_size")
        env = env[env.PC.cummax() == env.PC]        # record-breaking staircase
        ax.plot(env.candset_size, env.PC, c="tab:red", lw=1.6, marker="o", ms=3,
                zorder=3, label="envelope (best model per budget)")
        ax.set(xscale="log", xlabel="candidate pairs |C|", ylabel="PC (recall)",
               title=f"{dataset} — recall vs cost")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

        ax = axes[r][1]
        s = summary[summary.dataset == dataset]
        ax.scatter(s.faiss_s, s.nAP, s=28, c="tab:blue", edgecolor="k", linewidth=0.3)
        for _, row in s.nlargest(3, "nAP").iterrows():
            ax.annotate(row.model[:22], (row.faiss_s, row.nAP), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="faiss seconds (index + query)", ylabel="nAP",
               title=f"{dataset} — effectiveness vs time")
        ax.grid(alpha=0.3)
    fig.suptitle("Blocking sweep — recall/cost trade-off and time", y=1.0, fontsize=13)
    fig.tight_layout()
    plt.show()


def blocking_report(storage=STORAGE, experiment=BLOCKING_EXPERIMENT,
                    datasets=None, pc_target: float = PC_TARGET,
                    plots: bool = True) -> pd.DataFrame:
    """Full blocking-sweep view: per-dataset summary, cross-dataset leaderboard,
    and the trade-off plots. Returns the summary frame."""
    df = blocking_frame(storage, experiment, datasets)
    if df.empty:
        print(f"No '{experiment}' studies in {storage} — run the sweep first.")
        return df
    summary = blocking_summary(df, pc_target)
    print(f"=== {len(df)} cells over {df.dataset.nunique()} dataset(s), "
          f"{df.embedding_model.nunique()} models, "
          f"top_k <= {int(df.top_k.max())}, queried {sorted(set(df.queried_side))} ===")
    for dataset in dict.fromkeys(df.dataset):
        print(f"\n--- {dataset} (top 5 by AP; k* = smallest k reaching "
              f"PC >= {pc_target}) ---")
        display(summary[summary.dataset == dataset]
                .nlargest(5, "AP").reset_index(drop=True))
    print("\n=== cross-dataset leaderboard (best queried side per dataset) ===")
    display(blocking_leaderboard(summary))
    if plots:
        plot_blocking(df, summary)
    return summary
