import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import altair as alt
import os

# -------------------- GOOGLE SHEETS SETUP --------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

# -------------------- LOAD DATA --------------------
def get_acad_history_data(gc, class_name, term):
    sheet_name = f"{class_name} ACAD HISTORY"
    sheet = gc.open("FOXTROT DASHBOARD V2").worksheet(sheet_name)
    data = pd.DataFrame(sheet.get_all_records())
    if not data.empty:
        data = data[data['Term'] == term]
    return data

def load_all_history(term):
    data_all = []
    for cls in ["1CL", "2CL", "3CL"]:
        df = get_acad_history_data(client, cls, term)
        if not df.empty:
            df["Class"] = cls
            data_all.append(df)
    return pd.concat(data_all, ignore_index=True) if data_all else pd.DataFrame()

# -------------------- ACADEMIC SUMMARY --------------------
def show_deficiency_table(df):
    st.subheader("📉 Deficiency List (Grade < 7.0)")
    deficient = df[df['Grade'] < 7.0]
    if deficient.empty:
        st.info("No deficiencies recorded.")
        return
    st.dataframe(deficient[["Class", "Name", "Subject", "Grade"]].sort_values(by=["Class", "Subject", "Name"]))

    # Chart: Count of deficiencies per subject
    chart_data = deficient.groupby(["Class", "Subject"]).size().reset_index(name="Count")
    st.altair_chart(
        alt.Chart(chart_data).mark_bar().encode(
            x="Subject:N",
            y="Count:Q",
            color="Class:N",
            tooltip=["Class", "Subject", "Count"]
        ).properties(width=700).configure_axis(labelAngle=45),
        use_container_width=True
    )

def show_multiple_deficiencies(df):
    st.subheader("⚠️ Cadets with 2 or More Deficiencies")
    def_df = df[df["Grade"] < 7.0]
    count_df = def_df.groupby(["Class", "Name"]).size().reset_index(name="Deficiencies")
    multi_def = count_df[count_df["Deficiencies"] >= 2]
    if multi_def.empty:
        st.info("No cadets with multiple deficiencies.")
        return
    merged = pd.merge(multi_def, def_df, on=["Class", "Name"])
    st.dataframe(merged[["Class", "Name", "Subject", "Grade", "Deficiencies"]].sort_values(by=["Class", "Deficiencies"], ascending=[True, False]))

def show_top_performers(df):
    st.subheader("🏅 Top Performers per Subject")
    top_n = 3
    top_performers = df.groupby(["Class", "Subject"]).apply(
        lambda x: x.nlargest(top_n, "Grade")
    ).reset_index(drop=True)
    st.dataframe(top_performers[["Class", "Subject", "Name", "Grade"]].sort_values(by=["Class", "Subject", "Grade"], ascending=[True, True, False]))

def academic_summary_admin():
    st.header("Academic Summary")
    term = st.selectbox("Select Term", options=["1st Term", "2nd Term"], key="acad_term")
    acad_data = load_all_history(term)
    if acad_data.empty:
        st.warning("No academic history data available.")
        return
    show_deficiency_table(acad_data)
    st.markdown("---")
    show_multiple_deficiencies(acad_data)
    st.markdown("---")
    show_top_performers(acad_data)

# -------------------- PLACEHOLDER FOR OTHER REPORTS --------------------
def pft_summary_admin():
    st.header("PFT Summary")
    st.info("PFT summary report will go here.")

def military_summary_admin():
    st.header("Military Summary")
    st.info("Military summary report will go here.")

def conduct_summary_admin():
    st.header("Conduct Summary")
    st.info("Conduct summary report will go here.")

# -------------------- MAIN DASHBOARD FUNCTION --------------------
def show_main_dashboard():
    with st.spinner("Loading data..."):
        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

        with t1:
            pic, info = st.columns([1, 2])
            with pic:
                name_disp = st.session_state.get("name", "default")
                img_path = f"profile_pics/{name_disp}.jpg"
                st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
            with info:
                st.write("Cadet Info")  # Replace with actual display logic

# -------------------- DASHBOARD ROUTER --------------------
from summary_dashboard import summary_dashboard_main

if "role" not in st.session_state:
    st.warning("Not logged in.")
    st.stop()

if st.session_state.get("role") == "admin":
    st.sidebar.title("🛠 Admin Tools")
    admin_page = st.sidebar.radio("Select Admin View", ["Main Dashboard", "Summary Dashboard"])

    if admin_page == "Summary Dashboard":
        summary_dashboard_main()
    else:
        show_main_dashboard()
else:
    show_main_dashboard()
