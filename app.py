import streamlit as st
import os
from academics import show_academics_tab
from pft import show_pft_tab
from military import show_military_tab
from conduct import show_conduct_tab
from summary_dashboard import show_summary_dashboard

# -------------------- RUN DASHBOARD --------------------
def run_foxtrot_dashboard():
    # -- Set initial session variables --
    if "cls" not in st.session_state:
        st.session_state["cls"] = None
    if "name_disp" not in st.session_state:
        st.session_state["name_disp"] = ""
    if "role" not in st.session_state:
        st.session_state["role"] = "admin"  # Change as needed

    # -- Sidebar Summary Report Button (Admin only) --
    if st.session_state.get("role") == "admin":
        if st.sidebar.button("📊 Generate Summary Report"):
            st.session_state["show_summary"] = True
            st.rerun()

    # -- Summary Dashboard Trigger --
    if st.session_state.get("show_summary"):
        show_summary_dashboard()
        if st.sidebar.button("🔙 Back to Main Dashboard"):
            st.session_state["show_summary"] = False
            st.rerun()
        st.stop()

    # -- Login / Context Setup --
    if not st.session_state.get("cls") or not st.session_state.get("name_disp"):
        st.sidebar.info("Please identify yourself to access the dashboard.")
        st.session_state["cls"] = st.sidebar.selectbox("Select Class Level", ["1CL", "2CL", "3CL"])
        st.session_state["name_disp"] = st.sidebar.text_input("Enter your Display Name")

        if st.session_state["name_disp"]:
            st.session_state["name_clean"] = st.session_state["name_disp"].strip().lower().replace(" ", "_")

        st.warning("Please complete the information to proceed.")
        st.stop()

    # -- Assign local variables --
    cls = st.session_state["cls"]
    name_disp = st.session_state["name_disp"]
    name_clean = st.session_state["name_clean"]

    # -- Main Tabs --
    t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

    with t1:
        pic, info = st.columns([1, 2])
        with pic:
            img_path = f"profile_pics/{name_disp}.jpg"
            st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
        with info:
            st.write(f"**Cadet Name:** {name_disp}")
            st.write(f"**Class:** {cls}")

    with t2:
        show_academics_tab(cls, name_clean, name_disp)

    with t3:
        show_pft_tab(cls, name_clean, name_disp)

    with t4:
        show_military_tab(cls, name_clean, name_disp)

    with t5:
        show_conduct_tab(cls, name_clean, name_disp)

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    run_foxtrot_dashboard()
