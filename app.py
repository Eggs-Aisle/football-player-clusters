# app.py — Interactive PCA + Ward Clusters (Football Players)
# -----------------------------------------------------------
# Final consolidated app (15 clusters)
# - Upload or load Cleaned_Football_Player_Data.csv
# - Fixed preprocessing (drop ID/name/team/age/height/minutes/rating)
# - Standardize → PCA(2) → Ward (t=15)
# - Interactive scatter (PC1 vs PC2)
# - Single-cluster radar with adjustable number of features (6–12)
# - Cluster descriptions with your finalized role names (no player examples)
# - Export: Player Name → Player Team → Cluster Group (number text) → Player Type → metrics

import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

# ----------------------
# Utilities
# ----------------------
@st.cache_data(show_spinner=False)
def load_csv(default_path: str = "Cleaned_Football_Player_Data.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(default_path)
    except Exception:
        return pd.DataFrame()


def download_link(df: pd.DataFrame, filename: str = "players_pca_ward_clusters.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download clustered CSV</a>'
    return href


def safe_numeric_df(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    keep_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    num_df = keep_df.select_dtypes(include=["number"]).copy()
    return num_df


def run_pca_and_ward(num_df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(num_df.values)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    Z = linkage(pcs, method="ward")
    labels = fcluster(Z, t=15, criterion="maxclust")

    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=num_df.index)
    pca_df["Cluster"] = labels

    X_df = pd.DataFrame(X, columns=num_df.columns, index=num_df.index)
    profile = X_df.groupby(labels).mean().rename_axis("Cluster").reset_index()

    return pca_df, labels, explained, profile


# ----------------------
# App UI
# ----------------------
st.set_page_config(page_title="Player Clusters — PCA + Ward (15)", layout="wide")
st.title("Player Role Clusters")

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"]) 
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_csv()

if df.empty:
    st.warning("No data loaded. Upload a CSV or place Cleaned_Football_Player_Data.csv next to this app.")
    st.stop()

# Preprocess
drop_cols = ["Player ID", "Player Name", "Player Team", "Player Age", "Player Height", "Mins Played", "rating"]
num_df = safe_numeric_df(df, drop_cols=drop_cols)
if num_df.shape[1] < 3:
    st.error("Need at least 3 numeric features to run PCA and clustering.")
    st.stop()

# Run PCA + Ward
pca_df, labels, explained, profile = run_pca_and_ward(num_df)

# Attach identifiers for plot
tooltip_cols = [c for c in ["Player Name", "Player Team"] if c in df.columns]
plot_df = pd.concat([df[tooltip_cols].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

# Cluster names (finalized)
cluster_descriptions = {
    1: ("Creative/Scoring Winger", []),
    2: ("Elite Goal Scorer", []),
    3: ("Roaming Striker", []),
    4: ("Transitional Midfielder", []),
    5: ("Wide Supporter", []),
    6: ("Deep-Lying Passer", []),
    7: ("Attacking Engine", []),
    8: ("Secondary striker", []),
    9: ("Defensive Engine", []),
    10: ("Tempo Controler", []),
    11: ("Goalkeeper", []),
    12: ("Holding Defender", []),
    13: ("Creative Attacker", []),
    14: ("Hybrid (DM-CB) Defender", []),
    15: ("Ball-Playing Defender", [])
}

# Derived descriptive columns
plot_df["Cluster Group"] = plot_df["Cluster"].astype(int).astype(str)  # number text only
plot_df["Player Type"] = plot_df["Cluster"].map(lambda x: cluster_descriptions.get(x, ("Unknown", []))[0])

# Header metrics
colA, colB, colC = st.columns(3)
colA.metric("Players", value=len(plot_df))
colB.metric("PC1 Variance", value=f"{explained[0]*100:.1f}%")
colC.metric("PC2 Variance", value=f"{explained[1]*100:.1f}%")

# Scatter plot (PC space)
st.subheader("PC1 vs PC2 (colored by Cluster)")
fig = px.scatter(
    plot_df,
    x="PC1", y="PC2",
    color="Cluster",
    hover_data=tooltip_cols,
    height=650,
)
st.plotly_chart(fig, use_container_width=True)

# Cluster descriptions (names only)
st.subheader("Cluster Descriptions")
for cid, (ctype, _) in cluster_descriptions.items():
    st.markdown(f"### Cluster {cid}: {ctype}")

# Single-cluster radar with adjustable number of features
st.subheader("Cluster Role Radar")
radar_k = st.slider("Number of features on radar", 6, 12, 8, 1)
cluster_id = st.selectbox("Select a cluster", sorted(profile["Cluster"].unique()), index=0, key="single_radar")

prof = profile.set_index("Cluster")
row = prof.loc[cluster_id]
feats = row.abs().sort_values(ascending=False).head(radar_k).index.tolist()
vals = row[feats].values
labels = [f"{f} ({'+' if v>=0 else '-'})" for f, v in zip(feats, vals)]

radial_max = float(max(abs(vals))) if len(vals) else 1.0

fig_single = go.Figure()
fig_single.add_trace(go.Scatterpolar(r=[abs(v) for v in vals], theta=labels, fill='toself'))
fig_single.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, radial_max])), showlegend=False,
                         title=f"Cluster {cluster_id}: {cluster_descriptions.get(cluster_id, ('Unknown', []))[0]}")
st.plotly_chart(fig_single, use_container_width=True)

# Export (include dropped columns + new columns, hide PC1/PC2/Cluster)
st.subheader("Export")


# Base meta columns from the original df
meta_cols = []
for c in ["Player Name", "Player Team"]:
if c in df.columns:
meta_cols.append(c)


# Dropped-but-requested columns to include back in the export
restore_cols = ["Player ID", "Player Age", "Player Height", "Mins Played", "rating"]
restore_cols = [c for c in restore_cols if c in df.columns]


# New columns from plot_df
new_cols_df = plot_df[[c for c in ["Cluster Group", "Player Type"] if c in plot_df.columns]].reset_index(drop=True)


# Assemble left block: meta + restored dropped columns
left_block = df[meta_cols + restore_cols].reset_index(drop=True)


# Right block: all numeric performance features retained in num_df
right_block = num_df.reset_index(drop=True)


# Concatenate in desired order: meta/team → restored cols → new cols → metrics
export_df = pd.concat([left_block, new_cols_df, right_block], axis=1)


csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
label="Download clustered CSV",
data=csv_bytes,
file_name="players_pca_ward_clusters.csv",
mime="text/csv"
)
