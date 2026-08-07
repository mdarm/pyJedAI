"""Streamlit dashboard for assessing **blocking** quality in record space.

The sibling ``tsne_dashboard.py`` plots pair space (``|u-v|`` and the ZeroER
similarity vector) — the geometry a *matcher* sees. This one plots the space
k-NN blocking actually searches: individual records, with each gold pair drawn
as an edge that is either retrieved at the current ``k`` or missed.

The question it answers is "why was this pair missed", which the KPI tables in
``reporting.py`` can only pose. Click a record on the queried side and the panel
lists what the blocker returned *instead*, in rank order, with the gold partner
flagged wherever it landed.

Build the export first::

    python blocking_space.py

then, from this directory::

    streamlit run blocking_dashboard.py

Requires streamlit >= 1.35 for the click-to-select (``on_select``) API.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(HERE, "..", "outputs", "blocking_space")

BLUE = "#4D6BFE"        # retrieved / highlight
GRAY = "#9AA7BD"        # records
RED = "#E4572E"         # missed

st.set_page_config(page_title="ZeroER blocking space", layout="wide")


@st.cache_data
def load(name):
    return pd.read_parquet(os.path.join(EXPORT_DIR, f"{name}.parquet"))


if not os.path.exists(os.path.join(EXPORT_DIR, "records.parquet")):
    st.error(f"No export in {EXPORT_DIR}. Run `python blocking_space.py` first.")
    st.stop()

records, gold, neighbours, meta = (load(n) for n in
                                   ("records", "gold", "neighbours", "meta"))

# --- controls -------------------------------------------------------------
st.sidebar.title("Blocking space")
dataset = st.sidebar.selectbox("Dataset", sorted(meta.dataset.unique()))
models = sorted(meta.loc[meta.dataset == dataset, "model"].unique())
model = st.sidebar.selectbox("Embedding model", models)

m = meta[(meta.dataset == dataset) & (meta.model == model)].iloc[0]
side = st.sidebar.radio(
    "Queried side", ["right", "left"],
    help="Which collection is queried; the other is indexed. On a 1:N gold "
         "standard this caps recall by construction.")
k = st.sidebar.slider("top_k", 1, int(m.k_rank), 5)
edge_show = st.sidebar.radio("Gold pairs", ["missed only", "all", "retrieved only",
                                            "none"])

sel = (slice(None),)
rec = records[(records.dataset == dataset) & (records.model == model)
              & records.plotted].copy()
g = gold[(gold.dataset == dataset) & (gold.model == model)].copy()
nb = neighbours[(neighbours.dataset == dataset) & (neighbours.model == model)
                & (neighbours.side == side)]

rank = g[f"rank_{side}"].to_numpy()
g["retrieved"] = rank <= k

# --- KPIs, exact: ranks are over the full gold standard, not the sample ----
n_queried = int(m.n_right if side == "right" else m.n_left)
tp = int(g.retrieved.sum())
cand = k * n_queried
pc, pq = tp / len(g), tp / cand
rr = 1 - cand / (int(m.n_left) * int(m.n_right))
ceiling = min(1.0, k * n_queried / len(g))     # k*|queried| pairs vs |gold|

c = st.columns(5)
c[0].metric("PC (recall)", f"{pc:.4f}")
c[1].metric("PQ (precision)", f"{pq:.4f}")
c[2].metric("|C|", f"{cand:,}")
c[3].metric("RR", f"{rr:.5f}")
c[4].metric("PC ceiling", f"{ceiling:.4f}",
            help="k x |queried collection| / |gold|. Below 1 the queried side "
                 "cannot reach full recall at this k, whatever the model does.")
if ceiling < 1.0:
    st.warning(
        f"Querying **{side}** at k={k} can return at most {cand:,} pairs against "
        f"{len(g):,} gold — recall is capped at {ceiling:.3f} by cardinality, "
        f"not by the embedding. Try the other side.")

st.caption(f"{dataset} · {model} · {int(m.plotted_left):,}+{int(m.plotted_right):,} "
           f"of {int(m.n_left):,}+{int(m.n_right):,} records drawn "
           f"(all gold endpoints are kept; ranks and KPIs use the full data).")

# --- figure ---------------------------------------------------------------
xy = {(s, r): (x, y) for s, r, x, y in
      zip(rec.side, rec.row, rec.x, rec.y)}


def edge_trace(sub, colour, name):
    """One trace holding every edge, None-separated — far faster than one each."""
    xs, ys = [], []
    for lr, rr_ in zip(sub.left_row, sub.right_row):
        a, b = xy.get(("left", lr)), xy.get(("right", rr_))
        if a and b:
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
    return go.Scattergl(x=xs, y=ys, mode="lines", name=f"{name} ({len(sub)})",
                        line=dict(color=colour, width=1), opacity=0.55,
                        hoverinfo="skip")


fig = go.Figure()
for s, colour, symbol in (("left", GRAY, "circle"), ("right", "#C9D3E3", "diamond")):
    r = rec[rec.side == s]
    fig.add_trace(go.Scattergl(
        x=r.x, y=r.y, mode="markers", name=f"{s} records",
        marker=dict(size=5, color=colour, symbol=symbol, line_width=0),
        customdata=np.stack([r.side, r.row.astype(str), r.id], axis=-1),
        hovertemplate="%{customdata[0]} · id %{customdata[2]}<extra></extra>",
    ))
if edge_show in ("all", "retrieved only"):
    fig.add_trace(edge_trace(g[g.retrieved], BLUE, "retrieved"))
if edge_show in ("all", "missed only"):
    fig.add_trace(edge_trace(g[~g.retrieved], RED, "missed"))

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(
    height=680, font=dict(size=15, color="#1B2430"),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                title_text="", itemsizing="constant"),
)
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                        key=f"{dataset}:{model}:{side}:{k}:{edge_show}")

# --- clicked record -> what the blocker retrieved --------------------------
points = event.selection.points if (event and event.selection) else []
picked = []
for p in points:
    cn, pi = p.get("curve_number", 0), p.get("point_index", p.get("point_number"))
    if pi is None or cn >= len(fig.data):
        continue
    cd = getattr(fig.data[cn], "customdata", None)
    if cd is not None and pi < len(cd):
        picked.append((cd[pi][0], int(cd[pi][1])))

attrs = [c for c in records.columns
         if c not in ("dataset", "model", "side", "row", "id", "x", "y", "plotted")]
full = records[(records.dataset == dataset) & (records.model == model)]
by_row = {(s, r): row for (s, r), row in
          zip(zip(full.side, full.row), full.to_dict("records"))}
other = "left" if side == "right" else "right"

if not picked:
    st.info("Click a record to see what the blocker retrieved for it.")
for s, row in dict.fromkeys(picked):
    rec_row = by_row.get((s, row), {})
    st.markdown(f"#### {s} record `{rec_row.get('id', row)}`")
    if s != side:
        st.caption(f"This record is on the *indexed* side. Switch **Queried side** "
                   f"to `{s}` to see the neighbours it retrieves.")
        st.dataframe(pd.DataFrame({"attribute": attrs,
                                   "value": [rec_row.get(a, "") for a in attrs]}),
                     use_container_width=True, hide_index=True)
        continue

    partners = g[g[f"{s}_row"] == row]
    if len(partners):
        for _, gp in partners.iterrows():
            r_ = int(gp[f"rank_{side}"])
            ok = r_ <= k
            st.markdown(
                f"gold partner `{gp[f'{other}_id']}` — "
                + (f"retrieved at **rank {r_}** ✅" if ok else
                   (f"rank **{r_}** ❌ (outside k={k})" if r_ <= int(m.k_rank)
                    else f"**not in the top {int(m.k_rank)}** ❌")))
    else:
        st.caption("Not part of any gold pair — every retrieval for it is a "
                   "false candidate.")

    partner_rows = set(partners[f"{other}_row"])
    got = nb[nb.query_row == row].sort_values("rank")
    table = pd.DataFrame([{
        "rank": int(t.rank),
        "gold": "★" if t.target_row in partner_rows else "",
        "in |C|": "✓" if t.rank <= k else "",
        "id": by_row.get((other, int(t.target_row)), {}).get("id", t.target_row),
        **{a: by_row.get((other, int(t.target_row)), {}).get(a, "") for a in attrs},
    } for t in got.itertuples()])
    st.dataframe(table, use_container_width=True, hide_index=True)
