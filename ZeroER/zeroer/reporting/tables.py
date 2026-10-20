"""Database-backed results and plots shared by the experiment notebooks."""
from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from IPython.display import display

from zeroer.config import MATCHING_STORAGE, OUTPUT_DIR, STORAGE
from zeroer.data.benchmarks import DATASETS

DB = Path(OUTPUT_DIR) / "optuna.db"
MATCHING_DB = Path(OUTPUT_DIR) / "optuna_matching.db"


def trial_count(study_name, finished=False, db=MATCHING_DB):
    """Count trials without creating or updating a study."""
    condition = "AND t.state IN ('COMPLETE', 'FAIL', 'PRUNED')" if finished else ""
    return int(sql(f"""
        SELECT COUNT(*) AS count FROM trials t JOIN studies s USING (study_id)
        WHERE s.study_name = ? {condition}
    """, db=db, params=(study_name,)).iloc[0, 0])


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


# --------------------------------------------------------------------------
# Matching — the reference ladder. Lives in
# MATCHING_STORAGE, not STORAGE: the blocking sweep is closed, and every matching
# experiment runs on its fixed candidate sets.
# --------------------------------------------------------------------------

LADDER_EXPERIMENT = "ladder"


def _pair_f1(tp, predicted, n_gold):
    """Pair F1 against the full gold standard, from counts: 2tp / (|pred| + |gold|)."""
    return 2 * tp / (predicted + n_gold)


def ladder_frame(storage=MATCHING_STORAGE, experiment=LADDER_EXPERIMENT,
                 datasets=None) -> pd.DataFrame:
    """One tidy row per ladder fit, every metric recomputed from the stored counts.

    Beside ZeroER's own P / R / F1 each row carries what it has to be read
    between: ``accept_all_f1`` (every candidate declared a match), ``floor_f1``
    (the blocker's cosine + greedy one-to-one, no threshold), ``oracle_f1`` (a
    perfect matcher on these candidates) and ``floor_oracle_f1`` (the best any
    one-to-one mapping could do — below ``oracle_f1`` wherever the gold standard
    is not one-to-one, which is where that floor is an ablation, not a reference).
    Fits that raised are kept, with NaN scores and their ``error``.
    """
    rows = []
    for dataset in (datasets or DATASETS):
        try:
            study = optuna.load_study(study_name=f"{dataset}_{experiment}",
                                      storage=storage)
        except KeyError:
            continue                              # dataset not run yet
        rows += [dict(dataset=dataset, **t.params, **t.user_attrs)
                 for t in study.trials]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["PC"] = df.tp_blocking / df.n_gold
    df["PQ"] = df.tp_blocking / df.candset_size
    df["precision"] = df.tp / df.predicted
    df["recall"] = df.tp / df.n_gold
    df["f1"] = _pair_f1(df.tp, df.predicted, df.n_gold)
    df["accept_all_f1"] = _pair_f1(df.tp_blocking, df.candset_size, df.n_gold)
    df["floor_f1"] = _pair_f1(df.floor_tp, df.floor_predicted, df.n_gold)
    df["oracle_f1"] = _pair_f1(df.tp_blocking, df.tp_blocking, df.n_gold)
    df["floor_oracle_f1"] = _pair_f1(df.floor_oracle_tp, df.floor_oracle_tp, df.n_gold)
    return df.sort_values(["dataset", "queried_side", "top_k", "config"],
                          ignore_index=True)


def plot_ladder(df: pd.DataFrame, configs=("B-paper", "B-code")) -> None:
    """F1 along each nested-k curve, one panel per (dataset, queried side), with
    the two no-matcher floors and the oracle; the 19 named rungs are ringed."""
    curves = list(df.groupby(["dataset", "queried_side"], sort=False).groups)
    fig, axes = plt.subplots(2, -(-len(curves) // 2), squeeze=False,
                             figsize=(3.6 * -(-len(curves) // 2), 6.4))
    for ax, (dataset, side) in zip(axes.ravel(), curves):
        d = df[(df.dataset == dataset) & (df.queried_side == side)]
        cell = d.drop_duplicates("top_k").sort_values("top_k")
        ax.plot(cell.top_k, cell.oracle_f1, c="k", lw=0.8, ls=":", label="oracle")
        ax.plot(cell.top_k, cell.accept_all_f1, c="gray", lw=1, label="accept-all")
        ax.plot(cell.top_k, cell.floor_f1, c="gray", lw=1, ls="--", label="cosine 1:1")
        for config, colour in zip(configs, ("tab:red", "tab:blue")):
            g = d[d.config == config].sort_values("top_k")
            ax.plot(g.top_k, g.f1, c=colour, marker="o", ms=3, lw=1.4, label=config)
            ax.scatter(g.top_k[g.rung], g.f1[g.rung], s=60, facecolor="none",
                       edgecolor=colour, zorder=3)
        ax.set(title=f"{dataset} — query {side}", xlabel="k", ylabel="F1 (full gold)",
               ylim=(0, 1.02))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(curves):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=7, loc="center right")
    fig.suptitle("Reference ladder — string-only ZeroER between its floors and ceiling",
                 fontsize=13)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Matching — tuning studies (the Matching section of hyperparameter-tuning.ipynb)
# and their full-data runs (zeroer.ipynb). A trial is one configuration fitted on
# every named rung of a dataset; its value is the mean F1 over those rungs.
# --------------------------------------------------------------------------

def matching_trials(experiment, full=False, storage=MATCHING_STORAGE,
                    datasets=None) -> pd.DataFrame:
    """One row per trial of ``<dataset>_<experiment>[_full]``: its params, its
    value (``mean_f1``) and its scalar attributes. ``config_hash`` identifies a
    configuration across datasets — what donor selection joins on."""
    rows = []
    for dataset in (datasets or DATASETS):
        name = f"{dataset}_{experiment}" + ("_full" if full else "")
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        rows += [dict(dataset=dataset, number=t.number, mean_f1=t.values[0], **t.params,
                      **{k: v for k, v in t.user_attrs.items() if not k.endswith("rungs")})
                 for t in study.trials if t.values is not None]
    return pd.DataFrame(rows)


def matching_rungs(experiment, full=False, attr="rungs", storage=MATCHING_STORAGE,
                   datasets=None) -> pd.DataFrame:
    """The per-rung counts behind every trial, metrics recomputed; ``attr`` is
    ``"rungs"`` or, on a full run, ``"holdout_rungs"`` (the complementary half in each query direction)."""
    rows = []
    for dataset in (datasets or DATASETS):
        name = f"{dataset}_{experiment}" + ("_full" if full else "")
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        rows += [dict(dataset=dataset, number=t.number,
                      selection=t.user_attrs.get("selection"), **rung)
                 for t in study.trials if t.values is not None
                 for rung in t.user_attrs.get(attr, [])]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["f1"] = _pair_f1(df.tp, df.predicted, df.n_gold)
        df["accept_all_f1"] = _pair_f1(df.tp_blocking, df.candset_size, df.n_gold)
        df["oracle_f1"] = _pair_f1(df.tp_blocking, df.tp_blocking, df.n_gold)
    return df


def donor_selected(trials: pd.DataFrame) -> pd.DataFrame:
    """Per target dataset, the configuration the *other* datasets would pick: the
    highest mean of their ``mean_f1`` among the configurations *every* dataset has
    evaluated (the shared seed design — a donor's own TPE trials were never run on
    the target, so they cannot be looked up there). Uses no label of the target — though it
    does use the donors' — so it is the transferable counterpart of the target's
    own best trial, which is tuned on the target's gold."""
    wide = trials.pivot_table(index="config_hash", columns="dataset", values="mean_f1",
                              aggfunc="first").dropna()
    rows = []
    for target in wide.columns:
        donors = wide.drop(columns=target)
        if donors.empty:
            continue
        pick = donors.mean(axis=1).idxmax()
        rows.append(dict(dataset=target, config_hash=pick,
                         donor_mean_f1=donors.mean(axis=1)[pick],
                         mean_f1=wide.loc[pick, target],
                         own_best_f1=wide[target].max()))
    return pd.DataFrame(rows)


def sql(query, db=DB, params=()):
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as con:
        return pd.read_sql_query(query, con, params=params)


def study_status(experiment, db=MATCHING_DB, full=False):
    names = [f"{dataset}_{experiment}" + ("_full" if full else "") for dataset in DATASETS]
    placeholders = ",".join("?" for _ in names)
    return sql(f"""
        SELECT s.study_name AS study,
               SUM(CASE WHEN t.state = 'COMPLETE' THEN 1 ELSE 0 END) AS complete,
               SUM(CASE WHEN t.state = 'FAIL' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN t.state = 'WAITING' THEN 1 ELSE 0 END) AS queued
        FROM studies s LEFT JOIN trials t USING (study_id)
        WHERE s.study_name IN ({placeholders})
        GROUP BY s.study_name ORDER BY s.study_name
    """, db=db, params=names)


def show_table(frame):
    if frame.empty:
        print("No results stored for this experiment.")
    else:
        display(frame.round(4))


def replay_results(experiment):
    tables = []
    for attr, scope in (("rungs", "full"), ("holdout_rungs", "complementary half")):
        rungs = matching_rungs(experiment, full=True, attr=attr)
        if rungs.empty:
            continue
        table = rungs.groupby(["dataset", "selection"]).f1.mean().unstack("selection")
        bounds = (rungs.drop_duplicates(["dataset", "queried_side", "top_k"])
                  .groupby("dataset")[["accept_all_f1", "oracle_f1"]].mean())
        tables.append(table.join(bounds).assign(scope=scope).set_index("scope", append=True))
    return pd.concat(tables).sort_index() if tables else pd.DataFrame()


def treatments_frame(storage=MATCHING_STORAGE, datasets=None):
    """One row per stored fit, the metrics recomputed from the raw counts."""
    rows = []
    for dataset in (datasets or DATASETS):
        try:
            study = optuna.load_study(study_name="{}_collapse".format(dataset), storage=storage)
        except KeyError:
            continue
        rows += [dict(dataset=dataset, **t.params, **t.user_attrs) for t in study.trials]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["PQ"] = df.tp_blocking / df.candset_size
    df["PC"] = df.tp_blocking / df.n_gold
    df["precision"] = df.tp / df.predicted
    df["recall"] = df.tp / df.n_gold
    df["f1"] = 2 * df.tp / (df.predicted + df.n_gold)
    df["accept_all_f1"] = 2 * df.tp_blocking / (df.candset_size + df.n_gold)
    df["oracle_f1"] = 2 * df.PC / (1 + df.PC)
    return df
