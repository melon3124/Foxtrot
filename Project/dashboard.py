import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from utils.auth import login
from utils.gsheet import init_gsheets, sheet_df, clean_cadet_name_for_comparison
from utils.demographics import load_demographics, display_class_selector, display_cadet_buttons
from utils.tabs import show_tabs


# --- Page Config ---
st.set_page_config(
    page_title="Foxtrot CIS Dashboard",
    page_icon="🦊",
    layout="wide"
)

# --- Styling ---
st.markdown(
    """
    <style>
        body, .stApp { background: #1e0000; color: white; }
        .centered { display:flex; justify-content:center; gap:40px; margin-bottom:20px; }
        .stSelectbox, .stButton>button { width:300px !important; margin:auto; }
        h1 { text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
for key in ["auth_ok", "role", "username", "last_report_fetch"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "auth_ok" else None if key != "last_report_fetch" else 0

# --- Login Flow ---
if not st.session_state.auth_ok:
    login()

# --- Sidebar ---
st.sidebar.success(f"Logged in as **{st.session_state.username.upper()}** ({st.session_state.role})")
if st.sidebar.button("🔓 Logout"):
    for key in ["auth_ok", "role", "username"]:
        st.session_state[key] = None
    st.rerun()

# --- Google Sheets ---
SS = init_gsheets()

# --- Load Demographics ---
st.session_state.demo_df = load_demographics(sheet_df("DEMOGRAPHICS"))
demo_df = st.session_state.demo_df

# --- Cadet UI State ---
for key in ["mode", "selected_class", "selected_cadet_display_name", "selected_cadet_cleaned_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "mode" else "class"

# --- Main Title ---
st.markdown("<h1>🦊 Welcome to Foxtrot Company CIS</h1>", unsafe_allow_html=True)

# --- Class Selector ---
cls = display_class_selector()

# --- Cadet Buttons ---
if cls:
    display_cadet_buttons(demo_df, cls)

# --- Tabs: Display if a cadet is selected ---
name_clean = st.session_state.selected_cadet_cleaned_name
name_disp = st.session_state.selected_cadet_display_name
if name_clean:
    st.markdown(f"## Showing details for: {name_disp}")
    show_tabs(demo_df, name_clean, name_disp, cls, SS)
