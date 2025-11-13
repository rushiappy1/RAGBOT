import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os

st.set_page_config(page_title="RAG Diagnostics", layout="wide")

st.title("📊 RAGBot Diagnostics Dashboard")

log_path = Path("logs/retrieval_log.csv")

if not log_path.exists():
    st.warning("No retrieval logs found yet. Run some /ask queries first.")
    st.stop()

df = pd.read_csv(log_path)

col1, col2, col3 = st.columns(3)
col1.metric("Queries Logged", len(df))
col2.metric("Mean P@1", round(df["p@1"].mean(), 3))
col3.metric("Mean Recall@5", round(df["recall@5"].mean(), 3))

st.divider()

tab1, tab2, tab3 = st.tabs(["Retrieval Performance", "Chunk Stats", "Embedding Refresh"])

with tab1:
    st.subheader("Retrieval Distribution by Insurer")
    if "insurer" in df.columns:
        fig = px.box(df, x="insurer", y="p@1", title="P@1 per Insurer")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency Histogram")
    if "latency_ms" in df.columns:
        fig2 = px.histogram(df, x="latency_ms", nbins=30, title="Query Latency (ms)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Chunk Statistics (live from DB)")
    st.write("👉 Will query `SELECT COUNT(*), AVG(approx_tokens)` from Postgres later.")
    if Path("data/chunks").exists():
        chunk_files = list(Path("data/chunks").glob("*.jsonl"))
        st.write(f"Found {len(chunk_files)} chunk files")
        for f in chunk_files[:5]:
            st.text(f.name)

with tab3:
    st.subheader("Embedding File Update Times")
    embed_dir = Path("data/embeddings") if Path("data/embeddings").exists() else Path("data")
    file_info = []
    for f in embed_dir.rglob("*.pkl"):
        file_info.append({
            "file": f.name,
            "last_modified": pd.to_datetime(os.path.getmtime(f), unit="s")
        })
    if file_info:
        df_embed = pd.DataFrame(file_info)
        st.dataframe(df_embed.sort_values("last_modified", ascending=False))
    else:
        st.info("No embeddings found yet.")
