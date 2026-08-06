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
SATURATED_SPREAD = 0.10     # nAP range below which a dataset cannot rank models


def blocking_frame(storage=STORAGE, experiment=BLOCKING_EXPERIMENT,
                   datasets=None) -> pd.DataFrame:
    """One tidy row per swept cell, with the per-cell metrics recomputed.

    PC (pair completeness / recall), PQ (pair quality / precision), RR
    (reduction ratio) and their harmonic mean F_PC,RR all follow from the
    stored ``tp`` / ``candset_size`` / ``n_gold`` counts.
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
    df["F_PC_RR"] = 2 * df.PC * df.RR / (df.PC + df.RR)
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
    compares blocking cost at equal quality. ``F_max``/``k_F`` give the best
    PC-vs-RR operating point and the k it sits at.
    """
    out = []
    for (dataset, model, side), g in df.groupby(
            ["dataset", "embedding_model", "queried_side"], sort=False):
        g = g.sort_values("top_k")
        pc, pq = g.PC.to_numpy(), g.PQ.to_numpy()
        n_queried = int(g.n_right.iloc[0] if side == "right" else g.n_left.iloc[0])
        n_gold = int(g.n_gold.iloc[0])
        ap = _average_precision(pc, pq)
        hit = g[g.PC >= pc_target]
        row = dict(
            dataset=dataset, model=model, queried_side=side,
            AP=round(ap, 4),
            # AP is capped by min(1, n_gold/n_queried): k=1 retrieves only
            # n_queried pairs, and PQ never exceeds 1 — so when a dataset has
            # more matches than queried entities the ceiling is 1, not the
            # ratio. That ceiling differs per dataset and per queried side, so
            # rescale to [0, 1] before averaging across datasets.
            nAP=round(ap / min(1.0, n_gold / n_queried), 4),
            PC_max=round(float(pc[-1]), 4),
        )
        i_f = g.F_PC_RR.idxmax()
        row["F_max"] = round(float(g.F_PC_RR[i_f]), 4)
        row["k_F"] = int(g.top_k[i_f])
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

    Sides are compared on ``nAP``, not ``AP``: AP's ceiling
    ``min(1, n_gold/n_queried)`` differs between the two sides of the same
    dataset, so a max-AP pick structurally favours whichever side queries fewer
    entities regardless of quality (on dblp_scholar the ceilings are 1.000 left
    vs 0.083 right, so left always wins on AP while right is the better blocker).
    """
    best = (summary.sort_values("nAP", ascending=False)
                   .drop_duplicates(["dataset", "model"]))
    best = best.assign(rank=best.groupby("dataset").nAP.rank(ascending=False,
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
        rows = summary[summary.dataset == dataset]
        # how far apart the models actually are: a narrow spread, or many models
        # tied at the top, means this dataset carries little ranking signal and
        # its column should be discounted in the leaderboard's mean_rank
        spread = rows.nAP.max() - rows.nAP.min()
        tied = int((rows.groupby("model").nAP.max() == rows.nAP.max()).sum())
        tag = " — SATURATED" if spread < SATURATED_SPREAD else ""
        print(f"\n--- {dataset} (top 5 by nAP; k* = smallest k reaching "
              f"PC >= {pc_target}; nAP spread {spread:.3f}, "
              f"{tied}/{rows.model.nunique()} models tied at top{tag}) ---")
        display(rows.nlargest(5, "nAP").reset_index(drop=True))
    print("\n=== cross-dataset leaderboard (best queried side per dataset) ===")
    display(blocking_leaderboard(summary))
    if plots:
        plot_blocking(df, summary)
    return summary


# --------------------------------------------------------------------------
# Literature-ready KPIs
#
# Two protocols, both derived from the same sweep, because the field reports
# blocking two incompatible ways and a comparison quoting only one is
# under-determined (SC-Block Sec 6.3-6.4):
#
#   A. recall-threshold  -- PC/PQ at the smallest k reaching PC >= 0.90,
#      querying TableA. UniBlocker's protocol; the only surface on which our
#      numbers sit next to published ones unmodified.
#   B. cost-at-recall    -- |C| (and k, PQ, RR) at a recall target, per side.
#      SC-Block's protocol and the efficiency claim proper.
#
# AP/mAP is deliberately absent: it is definition-sensitive and at least one
# published mAP does not follow from its own stated formula, so it is not
# comparable across papers. Use nAP for internal ranking only.
# --------------------------------------------------------------------------

# Published reference values, transcribed from the papers. Protocol A rows are
# (PC, PQ, K) at the smallest K reaching PC >= 90%, querying TableA.
UNIBLOCKER_TABLE = {                      # UniBlocker Tables 3 and 4
    #                          DeepBlocker         Sudowoodo         STransformer        UniBlocker         Sparkly
    "fodors_zagats":         [(100.00, 21.01, 1), (99.11, 20.83, 1), (93.75,  9.85,  2), (100.00, 21.01, 1), (100.00, 21.01, 1)],
    "dblp_acm":              [( 97.39, 82.80, 1), (97.21, 82.65, 1), (95.28, 81.00,  1), ( 99.19, 84.33, 1), ( 98.74, 83.94, 1)],
    "dblp_scholar":          [( 90.35, 20.52, 9), (90.01,  2.16, 85), (91.49, 23.38, 8), ( 91.55, 31.19, 6), ( 92.16, 31.40, 6)],
    "abt_buy":               [( 90.06,  1.02, 90), (90.52, 15.31, 6), (90.52,  7.07, 13), ( 93.16, 31.51, 3), ( 92.80, 31.39, 3)],
    "amazon_googleproducts": [( 90.15,  1.12, 77), (90.23,  7.17, 12), (90.46,  6.64, 13), ( 90.38, 17.24, 5), ( 91.62, 17.48, 5)],
}
UNIBLOCKER_METHODS = ["DeepBlocker", "Sudowoodo", "STransformer", "UniBlocker", "Sparkly"]
UNIBLOCKER_PC_TARGET = 0.90       # the preset PC threshold their K column reaches
UNIBLOCKER_SIDE = "left"          # they query TableA; recoverable from their PQ

# NLSHBlock Table 2: F1 = harmonic mean of PC and PQ, each at a per-dataset
# recall target inherited from DL-Block, measured on test splits (3:1:1).
NLSH_TABLE = {                    # dataset -> (target, best baseline F1, NLSHBlock F1)
    "abt_buy":               (0.89, 62.6, 91.6),
    "amazon_googleproducts": (0.97, 13.5, 16.2),
    "dblp_acm":              (0.99, 65.0, 65.0),
    "dblp_scholar":          (0.97,  7.9,  7.9),
}

# Datasets whose models are too tightly bunched to rank (see blocking_report's
# spread/tie diagnostics); excluded from cross-dataset aggregates.
SATURATED_DATASETS = ("dblp_acm", "fodors_zagats")


def _at_target(g, target, by="candset_size"):
    """The cheapest row of ``g`` reaching ``PC >= target`` (None if unreachable).

    ``|C|`` is quantised by integer k, so the cheapest row is rarely unique;
    ties are broken on PQ so that a best-of-many selection can never be beaten
    by one of its own members.
    """
    hit = g[g.PC >= target]
    if hit.empty:
        return None
    return hit.sort_values([by, "PQ"], ascending=[True, False]).iloc[0]


def uniblocker_comparison(df, target=UNIBLOCKER_PC_TARGET, side=UNIBLOCKER_SIDE,
                          uniform_model=None) -> pd.DataFrame:
    """Protocol A -- our PC/PQ/k beside the published table, one row per dataset.

    ``ours`` is a *single* model applied to every dataset, the like-for-like
    against a one-model-all-datasets method; ``oracle`` is the best model per
    dataset, which is not a comparable number and is shown only as the gap that
    per-dataset selection would buy. ``uniform_model=None`` picks the uniform
    model automatically: fewest total k summed over datasets, PQ breaking ties.
    """
    side_df = df[df.queried_side == side]
    if uniform_model is None:
        scores = {}
        for m, g in side_df.groupby("embedding_model"):
            hits = [_at_target(gg, target) for _, gg in g.groupby("dataset")]
            if any(h is None for h in hits):
                continue
            scores[m] = (sum(int(h.top_k) for h in hits),
                         -sum(float(h.PQ) for h in hits))
        if not scores:
            return pd.DataFrame()
        uniform_model = min(scores, key=scores.get)

    rows = []
    for dataset, lit in UNIBLOCKER_TABLE.items():
        g = side_df[side_df.dataset == dataset]
        if g.empty:
            continue
        row = {"dataset": dataset}
        for name, (pc, pq, k) in zip(UNIBLOCKER_METHODS, lit):
            row[name] = f"{pc:.2f}/{pq:.2f} k={k}"
        ours = _at_target(g[g.embedding_model == uniform_model], target)
        best = _at_target(g, target)
        row["ours"] = (f"{ours.PC*100:.2f}/{ours.PQ*100:.2f} k={int(ours.top_k)}"
                       if ours is not None else "unreached")
        row["oracle"] = (f"{best.PC*100:.2f}/{best.PQ*100:.2f} k={int(best.top_k)}"
                         if best is not None else "unreached")
        # verdict against UniBlocker: smaller k wins; equal k decided on PQ
        upc, upq, uk = lit[UNIBLOCKER_METHODS.index("UniBlocker")]
        if ours is None:
            row["vs UniBlocker"] = "n/a"
        elif int(ours.top_k) < uk:
            row["vs UniBlocker"] = f"win (k {int(ours.top_k)} vs {uk})"
        elif int(ours.top_k) > uk:
            row["vs UniBlocker"] = f"loss (k {int(ours.top_k)} vs {uk})"
        else:
            d = ours.PQ * 100 - upq
            row["vs UniBlocker"] = "tie" if abs(d) < 0.05 else f"{d:+.1f} PQ at k={uk}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["uniform_model"] = uniform_model
    return out


def cost_at_recall(df, target=UNIBLOCKER_PC_TARGET) -> pd.DataFrame:
    """Protocol B -- smallest candidate set reaching ``target``, per dataset/side.

    ``|C|`` is the efficiency headline; it is quantised by integer k
    (``|C| = k x n_queried``), so models needing the same k tie exactly and
    ``PQ`` is the tie-breaker. ``RR`` is stated against ``‖E‖ = n_left x n_right``.
    """
    rows = []
    for (dataset, side), g in df.groupby(["dataset", "queried_side"]):
        r = _at_target(g, target)
        if r is None:
            continue
        n_tied = int((g[(g.top_k == r.top_k) & (g.PC >= target)]
                      .candset_size == r.candset_size).sum())
        rows.append(dict(
            dataset=dataset, side=side, k=int(r.top_k),
            C=int(r.candset_size), E=int(r.n_left) * int(r.n_right),
            PC=round(float(r.PC), 4), PQ=round(float(r.PQ), 4),
            RR=round(float(r.RR), 5), models_tied=n_tied,
            model=r.embedding_model, faiss_s=float(r.faiss_seconds),
        ))
    return pd.DataFrame(rows).sort_values(["dataset", "C"])


def nlsh_comparison(df) -> pd.DataFrame:
    """Protocol C -- F1(PC, PQ) at NLSHBlock's per-dataset recall targets.

    Their figures are on 3:1:1 *test splits* while ours are on the full tables,
    so ``‖E‖`` differs and the PQ half of each F1 is not strictly commensurable.
    Integer k also means we usually overshoot the target, which costs PQ.
    """
    d = df.copy()
    d["F1"] = 2 * d.PC * d.PQ / (d.PC + d.PQ)
    rows = []
    for dataset, (target, base, nlsh) in NLSH_TABLE.items():
        g = d[(d.dataset == dataset) & (d.PC >= target)]
        if g.empty:
            continue
        r = g.loc[g.F1.idxmax()]
        rows.append(dict(
            dataset=dataset, target=target, best_baseline=base, NLSHBlock=nlsh,
            ours=round(float(r.F1) * 100, 1), ours_PC=round(float(r.PC) * 100, 1),
            ours_PQ=round(float(r.PQ) * 100, 1), k=int(r.top_k),
            side=r.queried_side, model=r.embedding_model,
        ))
    return pd.DataFrame(rows)


def efficiency_leaderboard(df, target=UNIBLOCKER_PC_TARGET,
                           exclude=SATURATED_DATASETS) -> pd.DataFrame:
    """Rank models on ``|C|`` at ``target`` (PQ breaking the integer-k ties).

    Saturated datasets are excluded: they cannot separate models, so including
    them only dilutes the ranking. Models that miss the target on any remaining
    dataset are dropped rather than imputed.
    """
    keep = df[~df.dataset.isin(exclude)]
    rows = []
    for (dataset, model), g in keep.groupby(["dataset", "embedding_model"]):
        r = _at_target(g, target)
        if r is not None:
            rows.append(dict(dataset=dataset, model=model,
                             C=int(r.candset_size), PQ=float(r.PQ)))
    c = pd.DataFrame(rows)
    n = keep.dataset.nunique()
    c = c.groupby("model").filter(lambda g: g.dataset.nunique() == n)
    if c.empty:
        return c
    # rank on |C|, then PQ within the ties |C| cannot resolve
    c["rank"] = c.groupby("dataset").apply(
        lambda g: g[["C"]].assign(neg_pq=-g.PQ).apply(tuple, axis=1).rank(method="min")
    ).reset_index(level=0, drop=True)
    board = c.groupby("model").agg(mean_rank=("rank", "mean"),
                                   mean_C=("C", "mean"),
                                   mean_PQ=("PQ", "mean"),
                                   datasets=("dataset", "nunique"))
    return board.sort_values(["mean_rank", "mean_PQ"],
                             ascending=[True, False]).round(4).reset_index()


def literature_report(storage=STORAGE, experiment=BLOCKING_EXPERIMENT,
                      datasets=None, target=UNIBLOCKER_PC_TARGET) -> dict:
    """Every literature-ready KPI table in one call. Returns them keyed by protocol."""
    df = blocking_frame(storage, experiment, datasets)
    if df.empty:
        print(f"No '{experiment}' studies in {storage} — run the sweep first.")
        return {}
    ub = uniblocker_comparison(df, target)
    print(f"=== A. recall-threshold protocol (PC/PQ at smallest k reaching "
          f"PC >= {target:.0%}, querying TableA={UNIBLOCKER_SIDE}) ===")
    print(f"    ours = {ub.attrs.get('uniform_model')}, one model on every dataset; "
          f"oracle = best model per dataset (not comparable, shown as the "
          f"selection gap)")
    display(ub)
    print(f"\n=== B. cost at recall >= {target:.0%}: smallest |C|, per queried side ===")
    print("    |C| = k x n_queried, so models needing the same k tie exactly "
          "(models_tied); PQ breaks them")
    display(cost_at_recall(df, target))
    print("\n=== C. NLSHBlock protocol: F1(PC,PQ) at their per-dataset targets ===")
    print("    caveat: their figures are on 3:1:1 test splits, ours on full "
          "tables — the PQ halves are not strictly commensurable")
    display(nlsh_comparison(df))
    print(f"\n=== model ranking by |C| at PC >= {target:.0%} "
          f"(excludes {', '.join(SATURATED_DATASETS)} — cannot rank) ===")
    display(efficiency_leaderboard(df, target))
    print("\nNot measured here: embedding-creation time and index size. Any "
          "efficiency claim needs both — the models above span 384 to 5376 "
          "dimensions and 22M to 14B parameters.")
    return {"recall_threshold": ub, "cost_at_recall": cost_at_recall(df, target),
            "nlsh": nlsh_comparison(df),
            "leaderboard": efficiency_leaderboard(df, target)}
