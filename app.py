import streamlit as st
from auth import authenticate
from constants import *
from demographics import render_demographics
from academics import render_academics_tab
from pft import render_pft_tab
from military import render_military_tab
from conduct import render_conduct_tab
from sheets_utils import sheet_df


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


if st.session_state.role == "cadet":
    st.title(f"🎖️ Cadet Dashboard - Welcome {st.session_state.username.capitalize()}")

    # --- Load Data ---
    demo_df = sheet_df("DEMOGRAPHICS")

    # --- Ensure Cadet Selection State Exists ---
    if "selected_cadet_display_name" not in st.session_state:
        st.session_state["selected_cadet_display_name"] = None
    if "selected_cadet_cleaned_name" not in st.session_state:
        st.session_state["selected_cadet_cleaned_name"] = None

    # --- Class Selection Dropdown ---
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    initial_idx = ["", *classes.keys()].index(st.session_state.get("selected_class") or "")
    selected = st.selectbox("Select Class Level", ["", *classes.keys()], index=initial_idx)
    if selected != st.session_state.get("selected_class"):
        st.session_state.update({
            "mode": "class",
            "selected_class": selected,
            "selected_cadet_display_name": None,
            "selected_cadet_cleaned_name": None
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Class View Grid ---
    cls = st.session_state.get("selected_class")
    if st.session_state.get("mode") == "class" and cls:
        cadets = demo_df[demo_df["CLASS"] == cls]
        if cadets.empty:
            st.warning(f"No cadets for class {cls}.")
        else:
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            for i in range(0, len(cadets), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j >= len(cadets):
                        continue
                    name_display = cadets.iloc[i + j]["FULL NAME_DISPLAY"]
                    name_cleaned = cadets.iloc[i + j]["FULL NAME"]
                    with cols[j]:
                        if st.button(name_display, key=f"cadet_{name_cleaned}_{cls}"):
                            st.session_state.selected_cadet_display_name = name_display
                            st.session_state.selected_cadet_cleaned_name = name_cleaned
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Cadet Tabs View ---
        name_disp = st.session_state.selected_cadet_display_name
        name_clean = st.session_state.selected_cadet_cleaned_name
        if name_clean:
            row = demo_df[demo_df["FULL NAME"] == name_clean].iloc[0]
            st.markdown(f"## Showing details for: {name_disp}")
            t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

            with t1:
                render_demographics(demo_df, cls)

            with t2:
                render_academics_tab(
                    row,
                    name_disp,
                    name_clean,
                    cls,
                    acad_sheet_map,
                    acad_hist_map
                )

            with t3:
                render_pft_tab(
                    name_disp,
                    name_clean,
                    cls,
                    pft_sheet_map,
                    pft2_sheet_map
                )

            with t4:
                render_military_tab(
                    name_disp,
                    name_clean,
                    cls,
                    mil_sheet_map
                )

            with t5:
                render_conduct_tab(
                    name_disp,
                    name_clean,
                    cls,
                    conduct_sheet_map
                )
    else:
        st.info("Please select your class and mode first.")


