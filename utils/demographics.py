import streamlit as st
import pandas as pd

@st.cache_data(ttl=300)
def load_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if "CLASS" not in df.columns or "CADNAME" not in df.columns:
        st.warning("DEMOGRAPHICS sheet is missing required columns.")
        return pd.DataFrame()
    return df

def display_class_selector() -> str:
    demo_df = st.session_state.get("demo_df")
    if demo_df is None or demo_df.empty:
        return None

    available_classes = sorted(demo_df["CLASS"].dropna().unique())
    selected_class = st.selectbox("Select Class", available_classes, key="class_selector")

    st.session_state.selected_class = selected_class
    return selected_class

def display_cadet_buttons(demo_df: pd.DataFrame, cls: str):
    filtered = demo_df[demo_df["CLASS"] == cls]
    cadets = sorted(filtered["CADNAME"].dropna().unique())

    st.markdown("### Select Cadet:")
    cols = st.columns(4)
    for i, cadet in enumerate(cadets):
        if cols[i % 4].button(cadet):
            st.session_state.selected_cadet_display_name = cadet
            st.session_state.selected_cadet_cleaned_name = cadet.strip().upper()

