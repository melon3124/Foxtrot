import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import re
import unicodedata
import time
import json
import pygsheets
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- GLOBAL VARIABLES & FUNCTIONS ---
if st.session_state.get("pft_refresh_triggered"):
    del st.session_state["pft_refresh_triggered"]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "t3"

def clean_column_names(df):
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def evaluate_status(grade):
    try:
        val = float(grade)
        return "Proficient" if val >= 7 else "DEFICIENT"
    except:
        return "N/A"

def clean_cadet_name_for_comparison(name):
    return name.strip().upper()

def update_gsheet(sheet_name, df):
    try:
        client = get_gsheet_client()
        sh = client.open("FOXTROT DASHBOARD V2")
        worksheet = sh.worksheet_by_title(sheet_name)
        worksheet.clear()
        worksheet.set_dataframe(df, (1, 1))
        st.success("Sheet update successful.")
    except Exception as e:
        st.error(f"Failed to update sheet: {e}")

def get_gsheet_client():
    return pygsheets.authorize(service_account_info=st.secrets["google_service_account"])

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(
    st.secrets["google_service_account"],
    scopes=scopes
)
gc = gspread.authorize(credentials)
sh = gc.open("FOXTROT DASHBOARD V2")

def update_sheet(sheet_name, updated_df):
    try:
        worksheet = sh.worksheet(sheet_name)
        worksheet.clear()
        worksheet.update([updated_df.columns.values.tolist()] + updated_df.values.tolist())
    except Exception as e:
        st.error(f"❌ Failed to update Google Sheet '{sheet_name}': {e}")

if "last_report_fetch" not in st.session_state:
    st.session_state["last_report_fetch"] = 0
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "view" not in st.session_state:
    st.session_state.view = "main"

# --- Login Logic ---
if not st.session_state.auth_ok:
    st.title("🦊 Foxtrot CIS Login")
    username = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    USER_CREDENTIALS = {
        "admin": {
            "password": "admin",
            "role": "admin"
        },
        "cadet": {
            "password": "cadet",
            "role": "cadet"
        }
    }

    if login_btn:
        user = USER_CREDENTIALS.get(username)
        if user and user["password"] == pw:
            st.session_state.auth_ok = True
            st.session_state.role = user["role"]
            st.session_state.username = username
            st.success(f"✅ Logged in as {username.upper()} ({user['role'].upper()})")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.stop()

# --- Logged In ---
st.sidebar.success(f"Logged in as **{st.session_state.username.upper()}** ({st.session_state.role})")

if st.sidebar.button("🔓 Logout"):
    st.session_state.auth_ok = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.view = "main"
    st.rerun()

if st.session_state.role == "admin":
    st.sidebar.markdown("---")
    dashboard_mode = st.sidebar.radio(
        "Select Dashboard View",
        ("Main Dashboard", "Summary Dashboard")
    )
    if dashboard_mode == "Summary Dashboard":
        st.session_state["view"] = "summary"
    else:
        st.session_state["view"] = "main"
# --- CONFIG & GOOGLE SHEETS SETUP ---
st.set_page_config(
    page_title="Foxtrot CIS Dashboard",
    page_icon="🦊",
    layout="wide"
)
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
st.markdown("<h1>🦊 Welcome to Foxtrot Company CIS</h1>", unsafe_allow_html=True)

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

try:
    creds = Credentials.from_service_account_info(
        st.secrets["google_service_account"],
        scopes=scope
    )
    client = gspread.authorize(creds)
    SS = client.open("FOXTROT DASHBOARD V2")
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}")
    st.stop()

def normalize_column_name(col: str) -> str:
    if not isinstance(col, str):
        col = str(col)
    col = unicodedata.normalize("NFKD", col)
    col = col.replace("\xa0", "").replace("\u202f", "").replace("\u2009", "").replace(" ", "")
    col = re.sub(r'[^\w\-/ ]+', '', col).strip()
    return col.upper()

@st.cache_data(ttl=300)
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df

@st.cache_data(ttl=300)
def sheet_df(name: str) -> pd.DataFrame:
    try:
        worksheet = SS.worksheet(name)
        return clean_df(pd.DataFrame(worksheet.get_all_records()))
    except gspread.exceptions.WorksheetNotFound:
        st.warning(f"Worksheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching sheet '{name}' (raw: '{name}'): {e}")
        return pd.DataFrame()

def clean_cadet_name_for_comparison(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r'\s+', ' ', name).strip().upper()


# --- DEMOGRAPHICS ---
demo_df = sheet_df("DEMOGRAPHICS")
if demo_df.empty:
    st.error("Demographics sheet missing.")
    st.stop()

demo_df["FULL NAME"] = demo_df.apply(
    lambda r: clean_cadet_name_for_comparison(
        f"{r.get('FAMILY NAME','').strip()}, {r.get('FIRST NAME','').strip()} {r.get('MIDDLE NAME','').strip()} {r.get('EXTN','').strip()}"
    ), axis=1
)
demo_df["FULL NAME_DISPLAY"] = demo_df.apply(
    lambda r: f"{r.get('FAMILY NAME','').strip()}, {r.get('FIRST NAME','').strip()} {r.get('MIDDLE NAME','').strip()} {r.get('EXTN','').strip()}".strip(), axis=1
)

classes = {
    "1CL": (2, 27),
    "2CL": (30, 61),
    "3CL": (63, 104)
}
if "CLASS" not in demo_df.columns:
    demo_df["CLASS"] = ""
for cls, (start, end) in classes.items():
    demo_df.iloc[start-1:end, demo_df.columns.get_loc("CLASS")] = cls
    
# --- ADMIN-ONLY SUMMARY DASHBOARD ---
if st.session_state.view == "summary" and st.session_state.role == "admin":
    st.title("📊 Summary Dashboard")
    
    acad_hist_map = {
        "1st Term": {"1CL": "1CL ACAD HISTORY", "2CL": "2CL ACAD HISTORY", "3CL": "3CL ACAD HISTORY"},
        "2nd Term": {"1CL": "1CL ACAD HISTORY 2", "2CL": "2CL ACAD HISTORY 2", "3CL": "3CL ACAD HISTORY 2"}
    }
    mil_sheet_map = {
        "1st Term": {"1CL": "1CL MIL", "2CL": "2CL MIL", "3CL": "3CL MIL"},
        "2nd Term": {"1CL": "1CL MIL 2", "2CL": "2CL MIL 2", "3CL": "3CL MIL 2"}
    }
    pft_sheet_map = {
        "1st Term": {"1CL": "1CL PFT", "2CL": "2CL PFT", "3CL": "3CL PFT"},
        "2nd Term": {"1CL": "1CL PFT 2", "2CL": "2CL PFT 2", "3CL": "3CL PFT 2"}
    }
    conduct_sheet_map = {
        "1st Term": {"1CL": "1CL CONDUCT", "2CL": "2CL CONDUCT", "3CL": "3CL CONDUCT"},
        "2nd Term": {"1CL": "1CL CONDUCT 2", "2CL": "2CL CONDUCT 2", "3CL": "3CL CONDUCT 2"}
    }
    
    st.subheader("CAMP Performance Table")
    camp_data = []
    
    for cls in classes:
        acad_proficient = 0
        acad_deficient = 0
        mil_proficient = 0
        mil_deficient = 0
        pft_proficient = 0
        pft_deficient = 0
        conduct_proficient = 0
        conduct_deficient = 0
    
        acad_df = sheet_df(acad_hist_map.get("1st Term", {}).get(cls))
        if not acad_df.empty:
            acad_df["CURRENT GRADE"] = pd.to_numeric(acad_df.get("CURRENT GRADE", pd.Series()), errors="coerce")
            acad_proficient = acad_df["CURRENT GRADE"].apply(lambda x: 1 if x >= 7 else 0).sum()
            acad_deficient = acad_df["CURRENT GRADE"].apply(lambda x: 1 if x < 7 else 0).sum()
    
        mil_df = sheet_df(mil_sheet_map.get("1st Term", {}).get(cls))
        if not mil_df.empty:
            if cls == "1CL":
                mil_df["GRADE"] = pd.to_numeric(mil_df.get("GRADE", pd.Series()), errors="coerce")
                mil_proficient = mil_df["GRADE"].apply(lambda x: 1 if x >= 7 else 0).sum()
                mil_deficient = mil_df["GRADE"].apply(lambda x: 1 if x < 7 else 0).sum()
            elif cls == "2CL":
                mil_df["AS"] = pd.to_numeric(mil_df.get("AS", pd.Series()), errors="coerce")
                mil_df["NS"] = pd.to_numeric(mil_df.get("NS", pd.Series()), errors="coerce")
                mil_df["AFS"] = pd.to_numeric(mil_df.get("AFS", pd.Series()), errors="coerce")
                proficient_cadets = mil_df[(mil_df["AS"] >= 7) & (mil_df["NS"] >= 7) & (mil_df["AFS"] >= 7)]
                deficient_cadets = mil_df[(mil_df["AS"] < 7) | (mil_df["NS"] < 7) | (mil_df["AFS"] < 7)]
                mil_proficient = len(proficient_cadets)
                mil_deficient = len(deficient_cadets)
            elif cls == "3CL":
                mil_df["MS231"] = pd.to_numeric(mil_df.get("MS231", pd.Series()), errors="coerce")
                mil_proficient = mil_df["MS231"].apply(lambda x: 1 if x >= 7 else 0).sum()
                mil_deficient = mil_df["MS231"].apply(lambda x: 1 if x < 7 else 0).sum()
    
        pft_df = sheet_df(pft_sheet_map.get("1st Term", {}).get(cls))
        if not pft_df.empty:
            pft_grade_cols = ["PUSHUPS_GRADES", "SITUPS_GRADES", "PULLUPS_GRADES", "RUN_GRADES"]
            if all(col in pft_df.columns for col in pft_grade_cols):
                pft_df[pft_grade_cols] = pft_df[pft_grade_cols].apply(pd.to_numeric, errors='coerce')
                pft_df['AVG_GRADE'] = pft_df[pft_grade_cols].mean(axis=1)
                pft_proficient = pft_df['AVG_GRADE'][pft_df['AVG_GRADE'] >= 7].shape[0]
                pft_deficient = pft_df['AVG_GRADE'][pft_df['AVG_GRADE'] < 7].shape[0]

        conduct_df = sheet_df(conduct_sheet_map.get("1st Term", {}).get(cls))
        if not conduct_df.empty:
            conduct_df["MERITS"] = pd.to_numeric(conduct_df.get("MERITS", pd.Series()), errors="coerce")
            conduct_proficient = conduct_df["MERITS"].apply(lambda x: 1 if x >= 0 else 0).sum()
            conduct_deficient = conduct_df["MERITS"].apply(lambda x: 1 if x < 0 else 0).sum()
    
        camp_data.append({
            "Class": cls,
            "Academics (Proficient)": acad_proficient,
            "Academics (Deficient)": acad_deficient,
            "PFT (Proficient)": pft_proficient,
            "PFT (Deficient)": pft_deficient,
            "Military (Proficient)": mil_proficient,
            "Military (Deficient)": mil_deficient,
            "Conduct (Proficient)": conduct_proficient,
            "Conduct (Deficient)": conduct_deficient,
        })
    
    st.dataframe(pd.DataFrame(camp_data).set_index("Class"), use_container_width=True)
    
    st.markdown("---")
    
    t_acad, t_pft, t_mil, t_conduct = st.tabs(["📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])
    
    with t_acad:
        st.subheader("Academic Summary")
        term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_acad_term")
        cls_select = st.selectbox("Select Class", ["1CL", "2CL", "3CL"], key="summary_acad_class")
        
        st.markdown(f"#### {cls_select} Academics")
        acad_df = sheet_df(acad_hist_map.get(term, {}).get(cls_select))
        if not acad_df.empty:
            subjects = [col for col in acad_df.columns if col not in ["NAME", "CURRENT GRADE", "STATUS", "DEF/PROF POINTS"]]
            for subj in subjects:
                if subj in acad_df.columns:
                    acad_df[subj] = pd.to_numeric(acad_df[subj], errors="coerce")
                    
                    summary_data = []
                    
                    proficient_cadets = acad_df[acad_df[subj] >= 7]["NAME"].dropna().tolist()
                    for cadet in proficient_cadets:
                        summary_data.append({"Status": "✅ Proficient", "Cadet": cadet})
                        
                    deficient_cadets = acad_df[acad_df[subj] < 7]["NAME"].dropna().tolist()
                    for cadet in deficient_cadets:
                        summary_data.append({"Status": "❌ Deficient", "Cadet": cadet})
                        
                    def_prof_points_series = acad_df.get("DEF/PROF POINTS")
                    highest_deficiency = "N/A"
                    if def_prof_points_series is not None and not def_prof_points_series.empty:
                        numeric_points = pd.to_numeric(def_prof_points_series, errors='coerce').dropna()
                        highest_deficiency = numeric_points.max() if not numeric_points.empty else "N/A"
                    
                    st.markdown(f"**Subject: {subj}**")
                    
                    summary_table = pd.DataFrame(summary_data)
                    if not summary_table.empty:
                        st.dataframe(summary_table, hide_index=True, use_container_width=True)
                    else:
                        st.info("No cadets found for this subject.")
                    
                    st.write(f"**Highest Deficiency Points:** {highest_deficiency if pd.notna(highest_deficiency) else 'N/A'}")
                    st.markdown("---")

    with t_pft:
        st.subheader("PFT Summary")
        term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_pft_term")
        cls_select = st.selectbox("Select Class", ["1CL", "2CL", "3CL"], key="summary_pft_class")
        
        st.markdown(f"#### {cls_select} PFT")
        pft_df = sheet_df(pft_sheet_map.get(term, {}).get(cls_select))
        if not pft_df.empty:
            pft_grade_cols = ["PUSHUPS_GRADES", "SITUPS_GRADES", "PULLUPS_GRADES", "RUN_GRADES"]
            
            if all(col in pft_df.columns for col in pft_grade_cols):
                pft_df[pft_grade_cols] = pft_df[pft_grade_cols].apply(pd.to_numeric, errors='coerce')
                pft_df['PFT_AVG_GRADE'] = pft_df[pft_grade_cols].mean(axis=1)

                # Determine SMC cadets based on average grade
                smc_cadets = pft_df[pft_df['PFT_AVG_GRADE'] < 7]['NAME'].dropna().tolist()
                smc_data = [{"Status": "❌ Failed", "Cadet": c} for c in smc_cadets]
                st.write("**SMC (Failed) Cadets:**")
                if smc_data:
                    st.dataframe(pd.DataFrame(smc_data), hide_index=True)
                else:
                    st.write("None")
                
                # Determine strongest cadets based on average grade and gender
                if 'GENDER' in pft_df.columns:
                    pft_df['GENDER'] = pft_df['GENDER'].str.upper().str.strip().map({'M': 'MALE', 'F': 'FEMALE'})
                    
                    strongest_male = pft_df[pft_df['GENDER'] == 'MALE'].sort_values(by='PFT_AVG_GRADE', ascending=False).iloc[0] if not pft_df[pft_df['GENDER'] == 'MALE'].empty else None
                    strongest_female = pft_df[pft_df['GENDER'] == 'FEMALE'].sort_values(by='PFT_AVG_GRADE', ascending=False).iloc[0] if not pft_df[pft_df['GENDER'] == 'FEMALE'].empty else None
                    
                    strongest_data = []
                    if strongest_male is not None:
                        strongest_data.append({"Rank": "👑 Strongest Male", "Cadet": strongest_male['NAME'], "Average Grade": f"{strongest_male['PFT_AVG_GRADE']:.2f}"})
                    if strongest_female is not None:
                        strongest_data.append({"Rank": "👑 Strongest Female", "Cadet": strongest_female['NAME'], "Average Grade": f"{strongest_female['PFT_AVG_GRADE']:.2f}"})

                    st.write("**Strongest Cadets (Highest Average Grade):**")
                    if strongest_data:
                        st.dataframe(pd.DataFrame(strongest_data), hide_index=True)
                    else:
                        st.write("None")
                else:
                    st.warning("Could not determine strongest cadets due to missing 'GENDER' column in the PFT sheet.")
            else:
                st.warning("Could not determine SMC or strongest cadets due to missing PFT grade columns (PUSHUPS_GRADES, etc.)")
    
    with t_mil:
        st.subheader("Military Summary")
        term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_mil_term")
        cls_select = st.selectbox("Select Class", ["1CL", "2CL", "3CL"], key="summary_mil_class")

        st.markdown(f"#### {cls_select} Military")
        mil_df = sheet_df(mil_sheet_map.get(term, {}).get(cls_select))
        if not mil_df.empty:
            proficient = []
            deficient = []

            if cls_select == "1CL":
                mil_df["GRADE"] = pd.to_numeric(mil_df.get("GRADE", pd.Series()), errors="coerce")
                proficient = mil_df[mil_df["GRADE"] >= 7]["NAME"].dropna().tolist()
                deficient = mil_df[mil_df["GRADE"] < 7]["NAME"].dropna().tolist()
            elif cls_select == "2CL":
                mil_df["AS"] = pd.to_numeric(mil_df.get("AS", pd.Series()), errors="coerce")
                mil_df["NS"] = pd.to_numeric(mil_df.get("NS", pd.Series()), errors="coerce")
                mil_df["AFS"] = pd.to_numeric(mil_df.get("AFS", pd.Series()), errors="coerce")
                
                proficient = mil_df[
                    (mil_df["AS"] >= 7) & (mil_df["NS"] >= 7) & (mil_df["AFS"] >= 7)
                ]["NAME"].dropna().tolist()
                deficient = mil_df[
                    (mil_df["AS"] < 7) | (mil_df["NS"] < 7) | (mil_df["AFS"] < 7)
                ]["NAME"].dropna().tolist()
            elif cls_select == "3CL":
                mil_df["MS231"] = pd.to_numeric(mil_df.get("MS231", pd.Series()), errors="coerce")
                proficient = mil_df[mil_df["MS231"] >= 7]["NAME"].dropna().tolist()
                deficient = mil_df[mil_df["MS231"] < 7]["NAME"].dropna().tolist()
            
            proficient_data = [{"Status": "✅ Proficient", "Cadet": c} for c in proficient]
            deficient_data = [{"Status": "❌ Deficient", "Cadet": c} for c in deficient]
            
            st.write("**Proficient Cadets:**")
            if proficient_data:
                st.dataframe(pd.DataFrame(proficient_data), hide_index=True)
            else:
                st.write("None")
            
            st.write("**Deficient Cadets:**")
            if deficient_data:
                st.dataframe(pd.DataFrame(deficient_data), hide_index=True)
            else:
                st.write("None")

    with t_conduct:
        st.subheader("Conduct Summary")
        term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_conduct_term")
        cls_select = st.selectbox("Select Class", ["1CL", "2CL", "3CL"], key="summary_conduct_class")

        st.markdown(f"#### {cls_select} Conduct")
        conduct_df = sheet_df(conduct_sheet_map.get(term, {}).get(cls_select))
        reports_df = sheet_df("REPORTS")
        
        if not conduct_df.empty:
            touring_status_col = None
            for col in conduct_df.columns:
                if "TOURING" in col.upper():
                    touring_status_col = col
                    break
            
            if touring_status_col:
                touring_cadets = conduct_df[conduct_df[touring_status_col].astype(str).str.lower().str.contains("touring", na=False)]["NAME"].dropna().tolist()
                touring_data = [{"Status": "✈️ Touring", "Cadet": c} for c in touring_cadets]
                st.write("**Touring Cadets:**")
                if touring_data:
                    st.dataframe(pd.DataFrame(touring_data), hide_index=True)
                else:
                    st.write("None")
            else:
                st.warning("⚠️ 'touring status' column is missing from the conduct sheet. Cannot determine touring cadets.")
        
        if not reports_df.empty:
            reports_df["DEMERITS"] = pd.to_numeric(reports_df.get("DEMERITS", pd.Series()), errors="coerce")
            reports_df = reports_df.dropna(subset=['NAME'])
            demerits_per_cadet = reports_df.groupby("NAME")["DEMERITS"].sum()
            
            red_cadets = demerits_per_cadet[demerits_per_cadet >= 20].index.tolist()
            red_data = [{"Status": "🚨 On the Red", "Cadet": c, "Demerits": int(demerits_per_cadet[c])} for c in red_cadets]

            st.write("**Cadets with >= 20 Demerits (on the red):**")
            if red_data:
                st.dataframe(pd.DataFrame(red_data), hide_index=True)
            else:
                st.write("None")
# --- MAIN DASHBOARD (FOR BOTH ROLES) ---
elif st.session_state.view == "main":
    for key in ["mode", "selected_class", "selected_cadet_display_name", "selected_cadet_cleaned_name"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "mode" else "class"

    st.markdown('<div class="centered">', unsafe_allow_html=True)
    initial_idx = ["", *classes.keys()].index(st.session_state.selected_class or "")
    selected = st.selectbox("Select Class Level", ["", *classes.keys()], index=initial_idx)
    if selected != st.session_state.selected_class:
        st.session_state.update({"mode": "class", "selected_class": selected, "selected_cadet_display_name": None, "selected_cadet_cleaned_name": None})
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    cls = st.session_state.selected_class
    if st.session_state.mode == "class" and cls:
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
                    name_display = cadets.iloc[i+j]["FULL NAME_DISPLAY"]
                    name_cleaned = cadets.iloc[i+j]["FULL NAME"]
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
            t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

            with t1:
                def display_field(label, value):
                    """Displays a formatted label-value pair."""
                    display_label = str(label).replace("_", " ").title()
                    display_value = value if pd.notna(value) and str(value).strip() else 'N/A'
                    st.markdown(f"**{display_label}**\n\n{display_value}")
                    st.markdown("---")

                def create_section(title, fields_to_display, num_columns=2):
                    if not fields_to_display:
                        return
                    st.subheader(title, divider="red")
                    cols = st.columns(num_columns)
                    col_index = 0
                    for label, value in fields_to_display.items():
                        with cols[col_index % num_columns]:
                            display_field(label, value)
                        col_index += 1

                with st.container(border=True):
                    left_spacer, pic_col, right_spacer = st.columns([1, 1, 1])
                    with pic_col:
                        img_path = f"profile_pics/{name_disp}.jpg"
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Cadet {row.get('FAMILY NAME', '')}", use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/400x400.png?text=No+Photo", caption="Photo Not Available", use_container_width=True)
                    st.markdown(f"<h1 style='text-align: center; color: white;'>{name_disp}</h1>", unsafe_allow_html=True)
                    class_info = row.get('CLASS', 'N/A')
                    afpsn_info = row.get('AFPSN', 'N/A')
                    st.markdown(f"<h3 style='text-align: center; color: #ffcccc;'>{class_info} | AFPSN: {afpsn_info}</h3>", unsafe_allow_html=True)
                    
                    details = row.to_dict()
                    personal_keywords = ['DATE OF BIRTH', 'AGE', 'HEIGHT', 'WEIGHT', 'COURSE', 'RELIGION', 'ETHNICITY', 'GENDER']
                    contact_keywords = ['CONTACT', 'EMAIL', 'ADDRESS']
                    guardian_keywords = ['GUARDIAN']
                    keys_to_ignore = ['FULL NAME', 'FULL NAME_DISPLAY', 'CLASS', 'AFPSN', 'FAMILY NAME', 'FIRST NAME', 'MIDDLE NAME', 'EXTN']
                    personal_fields = {}
                    contact_fields = {}
                    guardian_fields = {}
                    additional_fields = {}
                    
                    for key, value in details.items():
                        if key in keys_to_ignore:
                            continue
                        upper_key = key.upper()
                        if any(keyword in upper_key for keyword in personal_keywords):
                            personal_fields[key] = value
                        elif any(keyword in upper_key for keyword in contact_keywords):
                            contact_fields[key] = value
                        elif any(keyword in upper_key for keyword in guardian_keywords):
                            guardian_fields[key] = value
                        else:
                            additional_fields[key] = value
                    
                    create_section("👤 Personal Details", personal_fields)
                    create_section("📞 Contact Information", contact_fields, num_columns=1)
                    create_section("👨‍👩‍👧 Guardian Information", guardian_fields)
                    create_section("📋 Additional Details", additional_fields)

            with t2:
                try:
                    if "selected_term" not in st.session_state:
                        st.session_state.selected_term = "1st Term"

                    term = st.radio(
                        "Select Term",
                        ["1st Term", "2nd Term"],
                        index=["1st Term", "2nd Term"].index(st.session_state.selected_term),
                        horizontal=True,
                        help="Choose academic term for grade comparison"
                    )
                    st.session_state.selected_term = term
                    
                    acad_sheet_map = {
                        "1CL": {"1st Term": "1CL ACAD", "2nd Term": "1CL ACAD 2"},
                        "2CL": {"1st Term": "2CL ACAD", "2nd Term": "2CL ACAD 2"},
                        "3CL": {"1st Term": "3CL ACAD", "2nd Term": "3CL ACAD 2"}
                    }
                    acad_hist_map = {
                        "1CL": {"1st Term": "1CL ACAD HISTORY", "2nd Term": "1CL ACAD HISTORY 2"},
                        "2CL": {"1st Term": "2CL ACAD HISTORY", "2nd Term": "2CL ACAD HISTORY 2"},
                        "3CL": {"1st Term": "3CL ACAD HISTORY", "2nd Term": "3CL ACAD HISTORY 2"}
                    }
                    possible_name_cols = ["NAME", "FULL NAME", "CADET NAME"]
                    cols_to_remove = ["PREVIOUS GRADE", "DEF/PROF POINTS", "CURRENT GRADE", "INCREASE/DECREASE"]

                    def find_name_column(df):
                        upper_cols = pd.Index([str(c).strip().upper() for c in df.columns])
                        for col in possible_name_cols:
                            if col.upper() in upper_cols:
                                return df.columns[upper_cols.get_loc(col.upper())]
                        return None
                    
                    def get_worksheet_by_name(name):
                        for ws in SS.worksheets():
                            if ws.title.strip().upper() == name.strip().upper():
                                return ws
                        raise Exception(f"Worksheet '{name}' not found.")
                    
                    def update_sheet_rows(data, headers, name_idx, edited_df, name_clean, name_disp, target_column):
                        updated_data = {
                            "GRADES": {},
                            "DEF/PROF POINTS": None
                        }
                        
                        for _, r in edited_df.iterrows():
                            updated_data["GRADES"][r["SUBJECT"]] = str(r["CURRENT GRADE"]) if pd.notna(r["CURRENT GRADE"]) else ""
                            if updated_data["DEF/PROF POINTS"] is None and "DEF/PROF POINTS" in r:
                                updated_data["DEF/PROF POINTS"] = str(r["DEF/PROF POINTS"]) if pd.notna(r["DEF/PROF POINTS"]) else ""

                        cadet_row = None
                        cadet_row_index = -1
                        for i, row in enumerate(data[1:], 1):
                            if clean_cadet_name_for_comparison(row[name_idx]) == name_clean:
                                cadet_row = row
                                cadet_row_index = i
                                break
                        
                        if cadet_row is None:
                            cadet_row = [""] * len(headers)
                            cadet_row[name_idx] = name_disp
                            data.append(cadet_row)
                            cadet_row_index = len(data) - 1
                        
                        for subject, grade in updated_data["GRADES"].items():
                            try:
                                subj_idx = headers.index(subject)
                            except ValueError:
                                headers.append(subject)
                                subj_idx = len(headers) - 1
                                for i in range(len(data)):
                                    data[i].extend([""] * (subj_idx - len(data[i]) + 1))
                            cadet_row[subj_idx] = grade

                        if updated_data["DEF/PROF POINTS"] is not None:
                            try:
                                points_idx = headers.index("DEF/PROF POINTS")
                            except ValueError:
                                headers.append("DEF/PROF POINTS")
                                points_idx = len(headers) - 1
                                for i in range(len(data)):
                                    data[i].extend([""] * (points_idx - len(data[i]) + 1))
                            cadet_row[points_idx] = updated_data["DEF/PROF POINTS"]

                        data[cadet_row_index] = cadet_row
                        
                        return data, headers

                    prev_df = sheet_df(acad_sheet_map.get(row['CLASS'], {}).get("1st Term"))
                    hist_df = sheet_df(acad_hist_map.get(row['CLASS'], {}).get(term))
                    hist_df_1st_term = sheet_df(acad_hist_map.get(row['CLASS'], {}).get("1st Term"))
                    hist_df_2nd_term = sheet_df(acad_hist_map.get(row['CLASS'], {}).get("2nd Term"))

                    if not hist_df.empty:
                        hist_name_col = find_name_column(hist_df)
                        if hist_name_col:
                            acad_data = hist_df[hist_df[hist_name_col].apply(clean_cadet_name_for_comparison) == name_clean].iloc[0].to_dict()
                            
                            hist_df_1st_term_row = hist_df_1st_term[hist_df_1st_term[find_name_column(hist_df_1st_term)].apply(clean_cadet_name_for_comparison) == name_clean]
                            hist_df_2nd_term_row = hist_df_2nd_term[hist_df_2nd_term[find_name_column(hist_df_2nd_term)].apply(clean_cadet_name_for_comparison) == name_clean]
                            
                            df_display = pd.DataFrame([acad_data]).transpose().reset_index()
                            df_display.columns = ["SUBJECT", "CURRENT GRADE"]
                            
                            if term == "2nd Term":
                                # Merge with 1st term data for comparison
                                if not hist_df_1st_term_row.empty:
                                    prev_grades = hist_df_1st_term_row.iloc[0]
                                    df_display['PREVIOUS GRADE'] = df_display['SUBJECT'].map(prev_grades)
                            
                            df_display["STATUS"] = df_display["CURRENT GRADE"].apply(evaluate_status)
                            
                            def get_def_prof_points(grades):
                                try:
                                    val = float(grades)
                                    if val < 7:
                                        return 1
                                    else:
                                        return 0
                                except:
                                    return 0
                            
                            df_display['DEF/PROF POINTS'] = df_display['CURRENT GRADE'].apply(get_def_prof_points)
                            
                            st.dataframe(df_display, hide_index=True)

                            def update_grades():
                                try:
                                    ws_name = acad_hist_map[row['CLASS']][term]
                                    ws = get_worksheet_by_name(ws_name)
                                    all_data = ws.get_all_values()
                                    headers = [normalize_column_name(h) for h in all_data[0]]
                                    name_idx = [i for i, h in enumerate(headers) if h in [c.upper() for c in possible_name_cols]][0]
                                    
                                    edited_data, edited_headers = update_sheet_rows(all_data, headers, name_idx, df_display, name_clean, name_disp, "CURRENT GRADE")
                                    
                                    new_df = pd.DataFrame(edited_data[1:], columns=edited_headers)
                                    ws.clear()
                                    ws.set_dataframe(new_df, (1,1))
                                    st.success("✅ Grades updated successfully!")
                                    st.cache_data.clear()
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Failed to update grades: {e}")
                                    
                            if st.session_state.role == "admin":
                                if st.button("Update Academic Grades"):
                                    update_grades()

                            
                        else:
                            st.warning("Could not find the cadet in the academic records.")
                    else:
                        st.info(f"No academic data found for {row['CLASS']} for {term}.")
                except Exception as e:
                    st.error(f"An error occurred in the Academics tab: {e}")

            with t3:
                try:
                    if "pft_term" not in st.session_state:
                        st.session_state.pft_term = "1st Term"
                        
                    pft_term_radio = st.radio(
                        "Select Term",
                        ["1st Term", "2nd Term"],
                        index=["1st Term", "2nd Term"].index(st.session_state.pft_term),
                        horizontal=True,
                        key="pft_term_select"
                    )
                    st.session_state.pft_term = pft_term_radio
                    
                    pft_sheet_map = {
                        "1CL": {"1st Term": "1CL PFT", "2nd Term": "1CL PFT 2"},
                        "2CL": {"1st Term": "2CL PFT", "2nd Term": "2CL PFT 2"},
                        "3CL": {"1st Term": "3CL PFT", "2nd Term": "3CL PFT 2"}
                    }
                    pft_df = sheet_df(pft_sheet_map.get(row['CLASS'], {}).get(st.session_state.pft_term))
                    pft_name_col = find_name_column(pft_df)
                    
                    if not pft_df.empty and pft_name_col:
                        cadet_pft = pft_df[pft_df[pft_name_col].apply(clean_cadet_name_for_comparison) == name_clean]
                        if not cadet_pft.empty:
                            cadet_pft = cadet_pft.iloc[0]
                            st.markdown("### PFT Results")
                            cols = st.columns(4)
                            pft_metrics = {
                                "PUSHUPS": ("PUSHUPS_GRADES", "PUSHUPS_SCORE"),
                                "SITUPS": ("SITUPS_GRADES", "SITUPS_SCORE"),
                                "PULLUPS": ("PULLUPS_GRADES", "PULLUPS_SCORE"),
                                "RUN": ("RUN_GRADES", "RUN_SCORE")
                            }
                            
                            data_to_display = []
                            for metric, (grade_col, score_col) in pft_metrics.items():
                                score = cadet_pft.get(score_col, "N/A")
                                grade = cadet_pft.get(grade_col, "N/A")
                                status = evaluate_status(grade)
                                data_to_display.append([metric.title(), score, grade, status])

                            pft_display_df = pd.DataFrame(data_to_display, columns=["Activity", "Score", "Grade", "Status"])
                            
                            gb = GridOptionsBuilder.from_dataframe(pft_display_df)
                            gb.configure_column("Grade", cellStyle=lambda params: {"color": "red"} if params.value < 7 else {"color": "green"})
                            gb.configure_column("Status", cellStyle=lambda params: {"color": "red"} if params.value == "DEFICIENT" else {"color": "green"})
                            
                            grid_options = gb.build()
                            
                            grid_response = AgGrid(
                                pft_display_df,
                                gridOptions=grid_options,
                                data_return_mode='AS_INPUT',
                                update_mode='MODEL_CHANGED',
                                fit_columns_on_grid_load=True,
                                theme='streamlit',
                                allow_unsafe_jscode=True,
                                enable_enterprise_modules=False,
                            )
                            
                            if st.session_state.role == "admin":
                                # Admin PFT Edit functionality (same as original, but I'll add the UI here)
                                st.markdown("---")
                                st.subheader("Admin PFT Grade Editor")
                                if pft_name_col:
                                    pft_data = cadet_pft.to_dict()
                                    new_scores = {}
                                    for metric, (grade_col, score_col) in pft_metrics.items():
                                        new_scores[score_col] = st.text_input(f"New {metric} Score", value=pft_data.get(score_col, ''), key=f'edit_{name_clean}_{score_col}')
                                        new_scores[grade_col] = st.text_input(f"New {metric} Grade", value=pft_data.get(grade_col, ''), key=f'edit_{name_clean}_{grade_col}')

                                    if st.button("Update PFT Records"):
                                        try:
                                            # Get the current worksheet
                                            ws_name = pft_sheet_map[row['CLASS']][st.session_state.pft_term]
                                            worksheet = get_worksheet_by_name(ws_name)
                                            
                                            # Find the row for the current cadet
                                            all_records = worksheet.get_all_records()
                                            df_records = pd.DataFrame(all_records)
                                            
                                            if not df_records.empty:
                                                df_records.columns = [normalize_column_name(c) for c in df_records.columns]
                                                pft_name_col_normalized = normalize_column_name(pft_name_col)
                                                cadet_row_idx = df_records[df_records[pft_name_col_normalized].apply(clean_cadet_name_for_comparison) == name_clean].index
                                                
                                                if not cadet_row_idx.empty:
                                                    row_index = cadet_row_idx[0]
                                                    
                                                    # Update the scores and grades with the new values
                                                    for score_col, grade_col in pft_metrics.values():
                                                        if normalize_column_name(score_col) in df_records.columns:
                                                            df_records.loc[row_index, normalize_column_name(score_col)] = new_scores.get(score_col)
                                                        if normalize_column_name(grade_col) in df_records.columns:
                                                            df_records.loc[row_index, normalize_column_name(grade_col)] = new_scores.get(grade_col)
                                                            
                                                    # Update the Google Sheet
                                                    update_sheet(ws_name, df_records)
                                                    st.success("✅ PFT records updated successfully!")
                                                    st.cache_data.clear()
                                                    st.session_state["pft_refresh_triggered"] = True
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Cadet not found in the PFT sheet.")
                                            else:
                                                st.error("❌ PFT sheet is empty.")
                                        except Exception as e:
                                            st.error(f"❌ Failed to update PFT records: {e}")

                        else:
                            st.warning("No PFT data found for this cadet.")
                    else:
                        st.info(f"No PFT data found for {row['CLASS']} for {st.session_state.pft_term}.")
                except Exception as e:
                    st.error(f"An error occurred in the PFT tab: {e}")

            with t4:
                try:
                    mil_sheet_map = {
                        "1CL": {"1st Term": "1CL MIL", "2nd Term": "1CL MIL 2"},
                        "2CL": {"1st Term": "2CL MIL", "2nd Term": "2CL MIL 2"},
                        "3CL": {"1st Term": "3CL MIL", "2nd Term": "3CL MIL 2"}
                    }
                    mil_term = st.radio("Select Term", ["1st Term", "2nd Term"], horizontal=True, key="mil_term")
                    mil_df = sheet_df(mil_sheet_map.get(row['CLASS'], {}).get(mil_term))
                    mil_name_col = find_name_column(mil_df)
                    
                    if not mil_df.empty and mil_name_col:
                        cadet_mil = mil_df[mil_df[mil_name_col].apply(clean_cadet_name_for_comparison) == name_clean]
                        if not cadet_mil.empty:
                            cadet_mil_dict = cadet_mil.iloc[0].to_dict()
                            
                            st.markdown("### Military Grades")
                            
                            mil_grade_cols = []
                            if row['CLASS'] == '1CL':
                                mil_grade_cols = ['GRADE']
                            elif row['CLASS'] == '2CL':
                                mil_grade_cols = ['AS', 'NS', 'AFS']
                            elif row['CLASS'] == '3CL':
                                mil_grade_cols = ['MS231']
                            
                            mil_data = []
                            for col in mil_grade_cols:
                                grade = cadet_mil_dict.get(col, 'N/A')
                                status = "Proficient" if pd.to_numeric(grade, errors='coerce') >= 7 else "DEFICIENT"
                                mil_data.append([col, grade, status])
                            
                            mil_df_display = pd.DataFrame(mil_data, columns=["Military Subject", "Grade", "Status"])
                            
                            gb = GridOptionsBuilder.from_dataframe(mil_df_display)
                            gb.configure_column("Grade", cellStyle=lambda params: {"color": "red"} if pd.to_numeric(params.value, errors='coerce') < 7 else {"color": "green"})
                            gb.configure_column("Status", cellStyle=lambda params: {"color": "red"} if params.value == "DEFICIENT" else {"color": "green"})
                            grid_options = gb.build()
                            
                            AgGrid(
                                mil_df_display,
                                gridOptions=grid_options,
                                data_return_mode='AS_INPUT',
                                update_mode='MODEL_CHANGED',
                                fit_columns_on_grid_load=True,
                                theme='streamlit',
                                allow_unsafe_jscode=True,
                                enable_enterprise_modules=False,
                            )
                            
                            if st.session_state.role == "admin":
                                st.markdown("---")
                                st.subheader("Admin Military Grade Editor")
                                new_mil_grades = {}
                                for col in mil_grade_cols:
                                    new_mil_grades[col] = st.text_input(f"New Grade for {col}", value=cadet_mil_dict.get(col, ''), key=f'edit_{name_clean}_{col}')
                                    
                                if st.button("Update Military Records"):
                                    try:
                                        ws_name = mil_sheet_map[row['CLASS']][mil_term]
                                        worksheet = get_worksheet_by_name(ws_name)
                                        all_records = worksheet.get_all_records()
                                        df_records = pd.DataFrame(all_records)
                                        
                                        if not df_records.empty:
                                            df_records.columns = [normalize_column_name(c) for c in df_records.columns]
                                            mil_name_col_normalized = normalize_column_name(mil_name_col)
                                            cadet_row_idx = df_records[df_records[mil_name_col_normalized].apply(clean_cadet_name_for_comparison) == name_clean].index
                                            
                                            if not cadet_row_idx.empty:
                                                row_index = cadet_row_idx[0]
                                                for col in mil_grade_cols:
                                                    col_normalized = normalize_column_name(col)
                                                    if col_normalized in df_records.columns:
                                                        df_records.loc[row_index, col_normalized] = new_mil_grades.get(col)
                                                
                                                update_sheet(ws_name, df_records)
                                                st.success("✅ Military records updated successfully!")
                                                st.cache_data.clear()
                                                st.rerun()
                                            else:
                                                st.error("❌ Cadet not found in the military sheet.")
                                        else:
                                            st.error("❌ Military sheet is empty.")
                                    except Exception as e:
                                        st.error(f"❌ Failed to update military records: {e}")

                        else:
                            st.warning("No military data found for this cadet.")
                    else:
                        st.info(f"No military data found for {row['CLASS']} for {mil_term}.")
                except Exception as e:
                    st.error(f"An error occurred in the Military tab: {e}")

            with t5:
                try:
                    conduct_sheet_map = {
                        "1CL": {"1st Term": "1CL CONDUCT", "2nd Term": "1CL CONDUCT 2"},
                        "2CL": {"1st Term": "2CL CONDUCT", "2nd Term": "2CL CONDUCT 2"},
                        "3CL": {"1st Term": "3CL CONDUCT", "2nd Term": "3CL CONDUCT 2"}
                    }
                    reports_df = sheet_df("REPORTS")
                    
                    st.markdown("### Demerits and Merits")
                    
                    if "conduct_term" not in st.session_state:
                        st.session_state.conduct_term = "1st Term"
                    
                    conduct_term = st.radio("Select Term", ["1st Term", "2nd Term"], horizontal=True, key="conduct_term")
                    
                    conduct_df = sheet_df(conduct_sheet_map.get(row['CLASS'], {}).get(conduct_term))
                    conduct_name_col = find_name_column(conduct_df)
                    
                    if not conduct_df.empty and conduct_name_col:
                        cadet_conduct = conduct_df[conduct_df[conduct_name_col].apply(clean_cadet_name_for_comparison) == name_clean]
                        if not cadet_conduct.empty:
                            cadet_conduct_dict = cadet_conduct.iloc[0].to_dict()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                merits = cadet_conduct_dict.get('MERITS', 0)
                                demerits = cadet_conduct_dict.get('DEMERITS', 0)
                                st.metric("Merits", merits)
                                st.metric("Demerits", demerits)
                                
                            with col2:
                                touring_status_col = None
                                for col in conduct_df.columns:
                                    if "TOURING" in col.upper():
                                        touring_status_col = col
                                        break
                                if touring_status_col:
                                    touring_status = cadet_conduct_dict.get(touring_status_col, 'N/A')
                                    st.markdown(f"**Touring Status:** {touring_status}")
                                else:
                                    st.warning("⚠️ Touring status column not found.")
                            
                            # Admin Edit Functionality
                            if st.session_state.role == "admin":
                                st.markdown("---")
                                st.subheader("Admin Conduct Editor")
                                with st.form("conduct_form"):
                                    new_merits = st.number_input("Update Merits", value=pd.to_numeric(merits, errors='coerce'), format="%d", key=f'edit_merits_{name_clean}')
                                    new_demerits = st.number_input("Update Demerits", value=pd.to_numeric(demerits, errors='coerce'), format="%d", key=f'edit_demerits_{name_clean}')
                                    new_touring_status = st.text_input("Update Touring Status", value=touring_status, key=f'edit_touring_{name_clean}')
                                    
                                    submitted = st.form_submit_button("Update Conduct Records")
                                    if submitted:
                                        try:
                                            ws_name = conduct_sheet_map[row['CLASS']][conduct_term]
                                            worksheet = get_worksheet_by_name(ws_name)
                                            all_records = worksheet.get_all_records()
                                            df_records = pd.DataFrame(all_records)
                                            
                                            if not df_records.empty:
                                                df_records.columns = [normalize_column_name(c) for c in df_records.columns]
                                                conduct_name_col_normalized = normalize_column_name(conduct_name_col)
                                                cadet_row_idx = df_records[df_records[conduct_name_col_normalized].apply(clean_cadet_name_for_comparison) == name_clean].index
                                                
                                                if not cadet_row_idx.empty:
                                                    row_index = cadet_row_idx[0]
                                                    
                                                    if normalize_column_name('MERITS') in df_records.columns:
                                                        df_records.loc[row_index, normalize_column_name('MERITS')] = new_merits
                                                    if normalize_column_name('DEMERITS') in df_records.columns:
                                                        df_records.loc[row_index, normalize_column_name('DEMERITS')] = new_demerits
                                                    if touring_status_col and normalize_column_name(touring_status_col) in df_records.columns:
                                                        df_records.loc[row_index, normalize_column_name(touring_status_col)] = new_touring_status
                                                        
                                                    update_sheet(ws_name, df_records)
                                                    st.success("✅ Conduct records updated successfully!")
                                                    st.cache_data.clear()
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Cadet not found in the conduct sheet.")
                                            else:
                                                st.error("❌ Conduct sheet is empty.")
                                        except Exception as e:
                                            st.error(f"❌ Failed to update conduct records: {e}")

                        else:
                            st.warning("No conduct data found for this cadet.")
                    else:
                        st.info(f"No conduct data found for {row['CLASS']} for {conduct_term}.")
                    
                    st.markdown("### Demerits Summary")
                    if not reports_df.empty:
                        reports_df["DATE"] = pd.to_datetime(reports_df.get("DATE"), errors='coerce')
                        reports_df = reports_df.dropna(subset=["DATE", "DEMERITS"])
                        reports_df["DEMERITS"] = pd.to_numeric(reports_df["DEMERITS"], errors='coerce')
                        
                        cadet_reports = reports_df[reports_df["NAME"].apply(clean_cadet_name_for_comparison) == name_clean]
                        
                        if not cadet_reports.empty:
                            cadet_reports = cadet_reports.sort_values(by="DATE", ascending=False)
                            st.dataframe(cadet_reports, hide_index=True)
                        else:
                            st.info("No demerit reports found for this cadet.")
                    else:
                        st.info("The 'REPORTS' sheet is empty.")
                except Exception as e:
                    st.error(f"An error occurred in the Conduct tab: {e}")
