"""Streamlit dashboard for the ZeroER t-SNE embedding-space export.

Reads the tidy parquet written by ``embedding_space.ipynb`` (one row per
candidate pair x feature space, with each attribute split into its left/right
value) and lets you pick a point on the t-SNE map to inspect the original
record pair column by column.

Run (from this directory, in the data-env venv)::

    streamlit run tsne_dashboard.py

Requires streamlit >= 1.35 for the click-to-select (``on_select``) API.
"""
import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(HERE, "..", "outputs", "tsne_notebook")
COMBINED = os.path.join(EXPORT_DIR, "tsne_all.parquet")

st.set_page_config(page_title="ZeroER t-SNE explorer", layout="wide")


@st.cache_data
def load(path):
    return pd.read_parquet(path)


def attr_names(df):
    """Attribute columns are the left_<attr> columns, minus the bookkeeping id."""
    return [c[len("left_"):] for c in df.columns
            if c.startswith("left_") and c != "left_id"]


# --- load -----------------------------------------------------------------
if os.path.exists(COMBINED):
    df = load(COMBINED)
else:
    parts = sorted(glob.glob(os.path.join(EXPORT_DIR, "tsne_*.parquet")))
    if not parts:
        st.error(f"No export found in {EXPORT_DIR}. Run embedding_space.ipynb first.")
        st.stop()
    df = pd.concat([load(p) for p in parts], ignore_index=True)

attrs = attr_names(df)

# --- controls -------------------------------------------------------------
st.sidebar.title("ZeroER t-SNE explorer")
dataset = st.sidebar.selectbox("Dataset", sorted(df["dataset"].unique()))

dsub = df[df["dataset"] == dataset]
methods = list(dict.fromkeys(dsub["method"]))                 # keep notebook order
# method short-code -> HF model id (or "manual sim-features"), for friendly labels
model_of = (dsub.drop_duplicates("method").set_index("method")["model_id"].to_dict()
            if "model_id" in df.columns else {})
method = st.sidebar.selectbox(
    "Embedding model (PLM)", methods,
    format_func=lambda m: f"{m} · {model_of[m]}" if model_of.get(m) else m,
)
show = st.sidebar.radio("Show", ["both", "matches only", "non-matches only"])

sub = df[(df["dataset"] == dataset) & (df["method"] == method)].reset_index(drop=True)
if show == "matches only":
    sub = sub[sub["label"] == 1].reset_index(drop=True)
elif show == "non-matches only":
    sub = sub[sub["label"] == 0].reset_index(drop=True)

sub["_row"] = range(len(sub))
sub["class"] = sub["label"].map({1: "match", 0: "non-match"})
dim = int(sub["dim"].iloc[0]) if len(sub) else 0

n_pos = int((sub["label"] == 1).sum())
model_label = model_of.get(method, method)
st.subheader(f"{dataset} · {model_label} ({dim}-D) — {n_pos} matches / "
             f"{len(sub) - n_pos} non-matches")
st.caption("Click a point (or box/lasso select) to inspect the original pair below.")

# --- plot -----------------------------------------------------------------
# DeepSeek palette: matches in the accent blue (the highlight), non-matches in
# a muted gray.
DEEPSEEK_BLUE = "#4D6BFE"
fig = px.scatter(
    sub, x="x", y="y", color="class",
    category_orders={"class": ["non-match", "match"]},
    color_discrete_map={"non-match": "#9AA7BD", "match": DEEPSEEK_BLUE},
    custom_data=["_row"], opacity=0.6, height=650,
)
fig.update_traces(marker=dict(size=7, line_width=0),
                  hovertemplate="(%{x:.1f}, %{y:.1f})<extra></extra>")
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(
    font=dict(size=16, color="#1B2430"),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
    hoverlabel=dict(font_size=15),
    legend=dict(
        title_text="", font=dict(size=17), itemsizing="constant",
        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        bgcolor="rgba(255,255,255,0.4)", bordercolor=DEEPSEEK_BLUE, borderwidth=1,
    ),
)

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                        key=f"{dataset}:{method}:{show}")

# --- selected-pair detail -------------------------------------------------
# Map each selected point back to its dataframe row via (curve_number,
# point_index) read off *our* figure's customdata — robust regardless of how
# Streamlit serialises the event's own customdata field.
points = event.selection.points if (event and event.selection) else []
rows = set()
for p in points:
    cn = p.get("curve_number", 0)
    pi = p.get("point_index", p.get("point_number"))
    if pi is None or cn >= len(fig.data):
        continue
    rows.add(int(fig.data[cn].customdata[pi][0]))
rows = sorted(rows)

if not rows:
    st.info("No point selected yet — click one on the map above.")
else:
    st.markdown(f"### {len(rows)} selected pair(s)")
    for r in rows:
        rec = sub.iloc[r]
        tag = "🟥 MATCH" if rec["label"] else "⬜ non-match"
        st.markdown(f"**{tag}** — left `{rec['left_id']}` ▸ right `{rec['right_id']}`")
        detail = pd.DataFrame({
            "attribute": attrs,
            "left":  [rec[f"left_{a}"] for a in attrs],
            "right": [rec[f"right_{a}"] for a in attrs],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)
