import streamlit as st
from auth import authenticate
from constants import *
from demographics import render_demographics
from academics import render_academics_tab
from pft import render_pft_tab
from military import render_military_tab
from conduct import render_conduct_tab
from sheets_utils import sheet_df

demo_df = sheet_df("DEMOGRAPHICS")
st.dataframe(demo_df)

# Authenticate
authenticate()
# 🎛 Sidebar: session info + logout
with st.sidebar:
    st.markdown("## 🔐 Session Info")
    st.write(f"👤 User: `{st.session_state.username}`")
    st.write(f"🧑‍💼 Role: `{st.session_state.role}`")

    if st.button("🚪 Logout"):
        for key in ["auth_ok", "username", "role"]:
            st.session_state.pop(key, None)
        st.success("Logged out.")
        st.rerun()

# 🧭 Main UI Based on Role
if st.session_state.role == "admin":
    st.title("🛠 Admin Dashboard")
    st.write("Welcome, Admin. You can manage cadets, update grades, and view summaries.")
    # Placeholder for admin tools
    st.info("⚙ Add your admin tabs and tools here.")

elif st.session_state.role == "cadet":
    st.title("📘 Cadet View")
    st.write("Welcome, Cadet. Here's your profile and performance dashboard.")
    # Placeholder for cadet view
    st.info("📋 Show cadet-specific info, grades, PFT results, etc.")

else:
    st.error("🚫 Unknown role. Contact admin.")


# Load data
demo_df = sheet_df("DEMOSHEET")
cls = st.session_state.get("selected_class")

if st.session_state.get("mode") == "class" and cls:
    row = render_demographics(demo_df, cls)
    if row is not None:
        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])
        with t1: st.write("Demographics already shown.")
        with t2: render_academics_tab(row, st.session_state.selected_cadet_display_name, st.session_state.selected_cadet_cleaned_name, cls, acad_sheet_map, acad_hist_map)
        with t3: render_pft_tab(st.session_state.selected_cadet_display_name, st.session_state.selected_cadet_cleaned_name, cls, pft_sheet_map, pft2_sheet_map)
        with t4: render_military_tab(st.session_state.selected_cadet_display_name, st.session_state.selected_cadet_cleaned_name, cls, mil_sheet_map)
        with t5: render_conduct_tab(st.session_state.selected_cadet_display_name, st.session_state.selected_cadet_cleaned_name, cls, conduct_sheet_map)

