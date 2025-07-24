import streamlit as st
import os

def render_demographics(demo_df, cls):
    cadets = demo_df[demo_df["CLASS"] == cls]
    if cadets.empty:
        st.warning(f"No cadets for class {cls}.")
        return

    st.markdown('<div class="centered">', unsafe_allow_html=True)
    for i in range(0, len(cadets), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j >= len(cadets): break
            name_display = cadets.iloc[i + j]["FULL NAME_DISPLAY"]
            name_cleaned = cadets.iloc[i + j]["FULL NAME"]
            with cols[j]:
                if st.button(name_display, key=f"cadet_{name_cleaned}_{cls}"):
                    st.session_state.selected_cadet_display_name = name_display
                    st.session_state.selected_cadet_cleaned_name = name_cleaned
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    name_disp = st.session_state.selected_cadet_display_name
    name_clean = st.session_state.selected_cadet_cleaned_name
    if name_clean:
        row = demo_df[demo_df["FULL NAME"] == name_clean].iloc[0]
        st.markdown(f"## Showing details for: {name_disp}")
        return row
    return None

