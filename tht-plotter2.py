# -----------------------------------------------------------------------------
# Streamlit app: ThT Aggregation Half‑time (t₁/₂) Plotter – enhanced version
# -----------------------------------------------------------------------------
# • Upload raw fluorescence data, subtract blanks, normalise, average replicates.
# • Interactive group mapping now supports unit entry (µM, %, or leave blank).
# • "Add / Update" clears the label + unit inputs automatically **without the
#   StreamlitAPIException** (uses a reset flag and rerun).
# • Choose which groups to display in the curves and half‑time plots.
# • If *all* visible groups lack a unit (i.e. labels are plain numbers), you can
#   reorder the half‑time scatter by ascending t₁/₂.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import csv

# ----------------------------- Page config & title ---------------------------
st.set_page_config(page_title="Half‑time Plot", layout="wide")  # must be first
st.title("ThT Aggregation Half‑time (t₁/₂) Analyzer")

# ----------------------------- Helper functions ------------------------------

def safe_rerun():
    """Call st.rerun() for Streamlit ≥ 1.27; fall back to experimental_rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def norm_series(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else np.zeros_like(x)


def calc_t_half(sub: pd.DataFrame, time_col: str) -> float:
    """Return time at first crossing of Norm ≥ 0.5 using linear interpolation."""
    sub_sorted = sub.sort_values(time_col).reset_index(drop=True)
    above = sub_sorted[sub_sorted["Norm"] >= 0.5]
    if above.empty:
        return np.nan  # never crosses

    idx = above.index[0]
    if idx == 0:
        return sub_sorted[time_col].iloc[0]  # starts above 0.5

    # Linear interpolation between the last point below 0.5 and first ≥ 0.5
    x0, x1 = sub_sorted["Norm"].iloc[idx - 1], sub_sorted["Norm"].iloc[idx]
    y0, y1 = sub_sorted[time_col].iloc[idx - 1], sub_sorted[time_col].iloc[idx]

    if x1 == x0:
        return y1  # plateau

    return y0 + (0.5 - x0) / (x1 - x0) * (y1 - y0)


def is_float(label: str) -> bool:
    try:
        float(label)
        return True
    except ValueError:
        return False

# ----------------------------- Sidebar: file upload -------------------------
st.sidebar.header("1. Upload data file")
uploaded_file = st.sidebar.file_uploader(
    "Choose a tab‑, comma‑, or space‑delimited file", type=["txt", "tsv", "csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a data file to begin.")
    st.stop()

# ----------------------------- Load & preview data --------------------------
raw_bytes = uploaded_file.getvalue()
sample_text = raw_bytes.decode("utf‑8", errors="ignore")[:2048]

try:
    dialect = csv.Sniffer().sniff(sample_text, delimiters="\t,; ")
    delimiter = dialect.delimiter
except Exception:
    delimiter = "\t" if "\t" in sample_text else ","

df = pd.read_csv(StringIO(raw_bytes.decode("utf‑8", errors="ignore")), sep=delimiter)

st.subheader("Raw data (first 5 rows)")
st.dataframe(df.head())

# ----------------------------- Time / blank / sample cols -------------------
all_cols = list(df.columns)

st.sidebar.header("2. Column selection & blank correction")

time_col = st.sidebar.selectbox(
    "Time column", options=all_cols, index=0, help="Column containing time points"
)

blank_cols = st.sidebar.multiselect(
    "Blank well columns",
    options=[c for c in all_cols if c != time_col],
    help="Select ≥ 1 wells that contain buffer (no sample)",
)

if len(blank_cols) == 0:
    st.error("Please select at least one blank column.")
    st.stop()

sample_cols = [c for c in all_cols if c not in blank_cols + [time_col]]

# ----------------------------- Group mapping --------------------------------
st.sidebar.header("3. Build group mapping")

# Persist mapping dict across reruns. Each value is the *display label* string.
if "mapping" not in st.session_state:
    st.session_state["mapping"] = {}

# Handle reset‑flag: clear inputs **before** widgets are rendered on this run
if st.session_state.get("_reset_inputs", False):
    st.session_state["new_label"] = ""
    st.session_state["unit_input"] = ""
    st.session_state["_reset_inputs"] = False

unique_locations = sorted(sample_cols)
unassigned = [loc for loc in unique_locations if loc not in st.session_state["mapping"]]

selected_locs = st.sidebar.multiselect("Select locations to label", options=unassigned)
new_label = st.sidebar.text_input("Label", key="new_label")
new_unit = st.sidebar.text_input("Unit (optional)", key="unit_input", placeholder="µM, %, etc. Leave blank for none")

col_add, col_reset = st.sidebar.columns(2)

with col_add:
    if st.button("➕ Add / Update"):
        if not selected_locs or new_label.strip() == "":
            st.warning("Select at least one location *and* enter a label.")
        else:
            unit = new_unit.strip() or None
            full_label = f"{new_label.strip()}" + (f" {unit}" if unit else "")
            for loc in selected_locs:
                st.session_state["mapping"][loc] = full_label
            # Flag that inputs should be cleared on the *next* run
            st.session_state["_reset_inputs"] = True
            safe_rerun()

with col_reset:
    if st.button("🗑️ Reset mapping"):
        st.session_state["mapping"].clear()
        # Also clear current inputs
        st.session_state["_reset_inputs"] = True
        safe_rerun()

st.sidebar.markdown("**Current mapping:**")
if st.session_state["mapping"]:
    st.sidebar.table(
        pd.DataFrame(list(st.session_state["mapping"].items()), columns=["Location", "Group"])
    )
else:
    st.sidebar.info("No mappings yet – add some above.")

if not st.session_state["mapping"]:
    st.info("Add at least one mapping in the sidebar to continue.")
    st.stop()

# ----------------------------- Group visibility & ordering ------------------
all_group_labels = sorted(set(st.session_state["mapping"].values()))

st.sidebar.header("4. Display options")
selected_groups = st.sidebar.multiselect(
    "Show groups", options=all_group_labels, default=all_group_labels
)

# Determine if all visible group labels are numeric (i.e. no unit appended)
labels_are_numeric = all(is_float(g.split()[0]) for g in selected_groups)
order_option = "Label order"
if labels_are_numeric:
    order_option = st.sidebar.radio("Half‑time order", ["Label order", "Ascending t₁/₂"])

# ----------------------------- Core calculations ----------------------------
# 1) Blank subtraction

df["blank_avg"] = df[blank_cols].mean(axis=1)

sub_df = df[[time_col] + sample_cols].copy()
sub_df[sample_cols] = sub_df[sample_cols].subtract(df["blank_avg"], axis=0)

# 2) Tidy & normalise

df_long = sub_df.melt(id_vars=time_col, var_name="Location", value_name="Fluorescence")
df_long["Fluorescence"] = df_long["Fluorescence"].clip(lower=0)

df_long["Norm"] = df_long.groupby("Location")["Fluorescence"].transform(norm_series)

# 3) Apply mapping → Group label

df_long["Group"] = df_long["Location"].map(st.session_state["mapping"]).fillna("Other")

# 4) Filter by selected groups

df_long = df_long[df_long["Group"].isin(selected_groups)]

# 5) Averaged curves

avg_df = (
    df_long.groupby([time_col, "Group"], as_index=False)
    .agg(Norm_mean=("Norm", "mean"), Norm_sd=("Norm", "std"))
)

# 6) Half‑time per well → summary

th_points = (
    df_long.groupby(["Group", "Location"], as_index=False)
    .apply(lambda g: pd.Series({"t_half": calc_t_half(g, time_col)}))
)

ht_summary = (
    th_points.groupby("Group", as_index=False)
    .agg(t_half_mean=("t_half", "mean"), t_half_sd=("t_half", "std"))
)

# Optional ordering by ascending t₁/₂ for numeric labels
if labels_are_numeric and order_option == "Ascending t₁/₂":
    order_seq = (
        ht_summary.sort_values("t_half_mean")[["Group"]].squeeze().tolist()
    )
    category_orders = {"Group": order_seq}
else:
    category_orders = None

# ----------------------------- Plot tabs ------------------------------------
curve_tab, half_tab = st.tabs(["Curves", "Half‑time t₁/₂"])

with curve_tab:
    fig_curve = px.line(
        avg_df,
        x=time_col,
        y="Norm_mean",
        color="Group",
        error_y="Norm_sd",
        title="Normalized aggregation curves (mean ± SD)",
        labels={time_col: "Time", "Norm_mean": "Normalized Fluorescence"},
        category_orders=category_orders,
    )
    fig_curve.update_layout(legend_title_text="Group")
    st.plotly_chart(fig_curve, use_container_width=True)

with half_tab:
    fig_half = px.scatter(
        ht_summary,
        x="Group",
        y="t_half_mean",
        error_y="t_half_sd",
        title="Half‑time (t₁/₂) by Group",
        labels={"t_half_mean": "t₁/₂"},
        category_orders=category_orders,
    )
    fig_half.update_layout(xaxis_title="", legend_title_text="Group")
    st.plotly_chart(fig_half, use_container_width=True)
