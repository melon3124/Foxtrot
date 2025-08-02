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
    
# --- SUMMARY DASHBOARD ---
if st.session_state.view == "summary":
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
                    proficient = acad_df[acad_df[subj] >= 7]["NAME"].dropna().tolist()
                    deficient = acad_df[acad_df[subj] < 7]["NAME"].dropna().tolist()
                    
                    def_prof_points_series = acad_df.get("DEF/PROF POINTS")
                    if def_prof_points_series is not None and not def_prof_points_series.empty:
                        numeric_points = pd.to_numeric(def_prof_points_series, errors='coerce').dropna()
                        highest_deficiency = numeric_points.max() if not numeric_points.empty else "N/A"
                    else:
                        highest_deficiency = "N/A"
                    
                    st.markdown(f"**Subject: {subj}**")
                    
                    # Create and display the table for proficient and deficient cadets
                    summary_table = pd.DataFrame({
                        "Status": ["Proficient", "Deficient"],
                        "Cadets": [', '.join(proficient) if proficient else "None", ', '.join(deficient) if deficient else "None"],
                        "Count": [len(proficient), len(deficient)]
                    })
                    st.dataframe(summary_table, hide_index=True, use_container_width=True)
                    
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
                st.write("**SMC (Failed) Cadets:**")
                if smc_cadets:
                    st.write(f"{', '.join(smc_cadets)}")
                else:
                    st.write("None")
                
                # Determine strongest cadets based on average grade and gender
                if 'GENDER' in pft_df.columns:
                    # Map 'M' and 'F' to 'MALE' and 'FEMALE'
                    pft_df['GENDER'] = pft_df['GENDER'].str.upper().str.strip().map({'M': 'MALE', 'F': 'FEMALE'})
                    
                    strongest_male = pft_df[pft_df['GENDER'] == 'MALE'].sort_values(by='PFT_AVG_GRADE', ascending=False).iloc[0] if not pft_df[pft_df['GENDER'] == 'MALE'].empty else None
                    strongest_female = pft_df[pft_df['GENDER'] == 'FEMALE'].sort_values(by='PFT_AVG_GRADE', ascending=False).iloc[0] if not pft_df[pft_df['GENDER'] == 'FEMALE'].empty else None
    
                    st.write("**Strongest Cadets (Highest Average Grade):**")
                    if strongest_male is not None:
                        st.write(f"**Male:** {strongest_male['NAME']} (Grade: {strongest_male['PFT_AVG_GRADE']:.2f})")
                    if strongest_female is not None:
                        st.write(f"**Female:** {strongest_female['NAME']} (Grade: {strongest_female['PFT_AVG_GRADE']:.2f})")
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
            
            st.write("**Proficient:**")
            if proficient:
                st.write(f"{', '.join(proficient)}")
            else:
                st.write("None")
            
            st.write("**Deficient:**")
            if deficient:
                st.write(f"{', '.join(deficient)}")
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
            # Find the 'touring status' column, being flexible with naming
            touring_status_col = None
            for col in conduct_df.columns:
                if "TOURING" in col.upper():
                    touring_status_col = col
                    break
            
            if touring_status_col:
                touring_cadets = conduct_df[conduct_df[touring_status_col].astype(str).str.lower().str.contains("touring", na=False)]["NAME"].dropna().tolist()
                st.write("**Touring Cadets:**")
                if touring_cadets:
                    st.write(f"{', '.join(touring_cadets)}")
                else:
                    st.write("None")
            else:
                st.warning("⚠️ 'touring status' column is missing from the conduct sheet. Cannot determine touring cadets.")
        
        if not reports_df.empty:
            reports_df["DEMERITS"] = pd.to_numeric(reports_df.get("DEMERITS", pd.Series()), errors="coerce")
            reports_df = reports_df.dropna(subset=['NAME'])
            demerits_per_cadet = reports_df.groupby("NAME")["DEMERITS"].sum()
            
            red_cadets = demerits_per_cadet[demerits_per_cadet >= 20].index.tolist()
            
            st.write("**Cadets with >= 20 Demerits (on the red):**")
            if red_cadets:
                st.write(f"{', '.join(red_cadets)}")
            else:
                st.write("None")

# --- MAIN DASHBOARD ---
else:
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

            
