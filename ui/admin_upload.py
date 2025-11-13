import streamlit as st
import os
import requests

st.title("Admin: Upload Documents for Ingestion")

uploaded_file = st.file_uploader("Choose a document (PDF, TXT)", type=["pdf", "txt"])

if uploaded_file:
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded to {save_path}")

    if st.button("Start Ingestion"):
        try:
            # Call your ingestion backend API or run ingestion script here
            # Or trigger ingestion via FastAPI endpoint if available
            # Example: requests.post("http://localhost:8000/ingest", files={"file": open(save_path, "rb")})
            # For now, just display message
            st.info("Ingestion triggered (implement actual ingestion call)")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")
