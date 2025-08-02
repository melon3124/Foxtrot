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

# --- GLOBAL SETTINGS & INITIALIZATION ---
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
        sh = client.open("FOXTROT DASHBOARD V2")  # Replace with your sheet name
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

st.sidebar.success(f"Logged in as **{st.session_state.username.upper()}** ({st.session_state.role})")

if st.sidebar.button("🔓 Logout"):
    st.session_state.auth_ok = False
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()

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
        st.error(f"Error fetching sheet '{name}': {e}")
        return pd.DataFrame()

def clean_cadet_name_for_comparison(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r'\s+', ' ', name).strip().upper()

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

for key in ["mode", "selected_class", "selected_cadet_display_name", "selected_cadet_cleaned_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "mode" else "class"

# =================================================================
#  ADMIN-SPECIFIC DASHBOARD AND LOGIC
# =================================================================

if st.session_state.role == "admin":
    st.sidebar.header("Admin Dashboard")
    admin_dashboard_selection = st.sidebar.radio(
        "Choose a Dashboard",
        ["Summary Dashboard", "Main Dashboard"],
        key="admin_selection"
    )
    
    if admin_dashboard_selection == "Summary Dashboard":
        st.title("Summary Dashboard")
        st.subheader("CAMP Performance Table")

        # --- Load all data for summary reports ---
        acad_dfs = {cls: sheet_df(f"{cls} ACAD") for cls in classes}
        pft_dfs = {cls: sheet_df(f"{cls} PFT") for cls in classes}
        mil_dfs = {cls: sheet_df(f"{cls} MIL") for cls in classes}
        conduct_dfs = {cls: sheet_df(f"{cls} CONDUCT") for cls in classes} # Assuming sheet named "CONDUCT"
        
        # --- Generate CAMP Performance Table ---
        summary_data = []
        for cls in classes:
            acad_df = acad_dfs.get(cls)
            pft_df = pft_dfs.get(cls)
            mil_df = mil_dfs.get(cls)
            conduct_df = conduct_dfs.get(cls)
            
            # Academics Summary (Assuming grade column)
            acad_prof = acad_df[acad_df["AVERAGE GRADE"] >= 7].shape[0] if not acad_df.empty and "AVERAGE GRADE" in acad_df.columns else 0
            acad_def = acad_df[acad_df["AVERAGE GRADE"] < 7].shape[0] if not acad_df.empty and "AVERAGE GRADE" in acad_df.columns else 0

            # PFT Summary (Assuming "PFT STATUS" column)
            pft_prof = pft_df[pft_df["PFT STATUS"] == "Pass"].shape[0] if not pft_df.empty and "PFT STATUS" in pft_df.columns else 0
            pft_def = pft_df[pft_df["PFT STATUS"] == "SMC"].shape[0] if not pft_df.empty and "PFT STATUS" in pft_df.columns else 0
            
            # Military Summary (Assuming "STATUS" column)
            mil_prof = mil_df[mil_df["STATUS"] == "Proficient"].shape[0] if not mil_df.empty and "STATUS" in mil_df.columns else 0
            mil_def = mil_df[mil_df["STATUS"] == "Deficient"].shape[0] if not mil_df.empty and "STATUS" in mil_df.columns else 0

            # Conduct Summary (Assuming "DEMERITS" column)
            conduct_prof = conduct_df[conduct_df["DEMERITS"] < 20].shape[0] if not conduct_df.empty and "DEMERITS" in conduct_df.columns else 0
            conduct_def = conduct_df[conduct_df["DEMERITS"] >= 20].shape[0] if not conduct_df.empty and "DEMERITS" in conduct_df.columns else 0

            summary_data.append({
                "CLASS": cls,
                "Academic Proficient": acad_prof,
                "Academic Deficient": acad_def,
                "PFT Proficient": pft_prof,
                "PFT Deficient": pft_def,
                "Military Proficient": mil_prof,
                "Military Deficient": mil_def,
                "Conduct Proficient": conduct_prof,
                "Conduct Deficient": conduct_def
            })
            
        camp_performance_table = pd.DataFrame(summary_data).set_index("CLASS")
        st.dataframe(camp_performance_table, use_container_width=True)

        # --- TABS FOR DETAILED REPORTS ---
        st.subheader("Detailed Summary Reports")
        tab1, tab2, tab3, tab4 = st.tabs(["Academics", "PFT", "Military", "Conduct"])

        with tab1:
            st.subheader("Academics Reports")
            for cls in classes:
                df = acad_dfs.get(cls)
                if df.empty or "AVERAGE GRADE" not in df.columns:
                    st.warning(f"No valid academic data for {cls} found.")
                    continue
                
                st.markdown(f"#### **{cls}**")
                
                proficient_cadets = df[df["AVERAGE GRADE"] >= 7][["NAME", "AVERAGE GRADE"]]
                st.markdown(f"**Proficient Cadets ({len(proficient_cadets)})**")
                st.dataframe(proficient_cadets, hide_index=True, use_container_width=True)
                
                deficient_cadets = df[df["AVERAGE GRADE"] < 7][["NAME", "AVERAGE GRADE", "DEFICIENCY POINTS"]]
                st.markdown(f"**Deficient Cadets ({len(deficient_cadets)})**")
                st.dataframe(deficient_cadets, hide_index=True, use_container_width=True)
                
                # Highest Deficiency Points (per subject, which requires more specific data)
                # Assuming 'DEFICIENCY POINTS' column exists and is numeric
                if "DEFICIENCY POINTS" in df.columns:
                    highest_deficiency = df.sort_values(by="DEFICIENCY POINTS", ascending=False).head(5)
                    st.markdown("**Highest Deficiency Points**")
                    st.dataframe(highest_deficiency[["NAME", "DEFICIENCY POINTS"]], hide_index=True, use_container_width=True)
                
        with tab2:
            st.subheader("PFT Reports")
            pft_dfs_2 = {cls: sheet_df(f"{cls} PFT 2") for cls in classes} # Assuming PFT 2 sheets exist
            
            for cls in classes:
                df1 = pft_dfs.get(cls)
                df2 = pft_dfs_2.get(cls)
                combined_df = pd.concat([df1, df2], ignore_index=True)
                
                if combined_df.empty or "PFT STATUS" not in combined_df.columns:
                    st.warning(f"No valid PFT data for {cls} found.")
                    continue
                    
                st.markdown(f"#### **{cls}**")
                
                # SMC Cadets
                smc_cadets = combined_df[combined_df["PFT STATUS"] == "SMC"].drop_duplicates(subset="NAME")
                st.markdown(f"**SMC Cadets ({len(smc_cadets)})**")
                st.dataframe(smc_cadets[["NAME"]], hide_index=True, use_container_width=True)
                
                # Strongest Cadets (Male and Female)
                if "PFT AVERAGE GRADE" in combined_df.columns and "GENDER" in combined_df.columns:
                    combined_df["PFT AVERAGE GRADE"] = pd.to_numeric(combined_df["PFT AVERAGE GRADE"], errors='coerce')
                    strongest_male = combined_df[(combined_df["GENDER"] == "Male") & (combined_df["PFT STATUS"] != "SMC")].sort_values(by="PFT AVERAGE GRADE", ascending=False).drop_duplicates(subset="NAME").head(1)
                    strongest_female = combined_df[(combined_df["GENDER"] == "Female") & (combined_df["PFT STATUS"] != "SMC")].sort_values(by="PFT AVERAGE GRADE", ascending=False).drop_duplicates(subset="NAME").head(1)

                    st.markdown("**Strongest Cadets (Highest Average Grade)**")
                    st.markdown("#### Males")
                    st.dataframe(strongest_male[["NAME", "PFT AVERAGE GRADE"]], hide_index=True, use_container_width=True)
                    st.markdown("#### Females")
                    st.dataframe(strongest_female[["NAME", "PFT AVERAGE GRADE"]], hide_index=True, use_container_width=True)

        with tab3:
            st.subheader("Military Reports")
            for cls in classes:
                df = mil_dfs.get(cls)
                if df.empty or "STATUS" not in df.columns:
                    st.warning(f"No valid military data for {cls} found.")
                    continue
                    
                st.markdown(f"#### **{cls}**")
                
                proficient_military = df[df["STATUS"] == "Proficient"]
                st.markdown(f"**Proficient Cadets ({len(proficient_military)})**")
                st.dataframe(proficient_military[["NAME"]], hide_index=True, use_container_width=True)
                
                deficient_military = df[df["STATUS"] == "Deficient"]
                st.markdown(f"**Deficient Cadets ({len(deficient_military)})**")
                st.dataframe(deficient_military[["NAME"]], hide_index=True, use_container_width=True)
                
        with tab4:
            st.subheader("Conduct Reports")
            for cls in classes:
                df = conduct_dfs.get(cls)
                if df.empty or "DEMERITS" not in df.columns or "TOUR" not in df.columns:
                    st.warning(f"No valid conduct data for {cls} found.")
                    continue
                    
                st.markdown(f"#### **{cls}**")
                
                # Touring Cadets
                touring_cadets = df[df["TOUR"] == "TRUE"]
                st.markdown(f"**Touring Cadets ({len(touring_cadets)})**")
                st.dataframe(touring_cadets[["NAME", "DEMERITS"]], hide_index=True, use_container_width=True)
                
                # Cadets with <20 Demerits ("on the red")
                df["DEMERITS"] = pd.to_numeric(df["DEMERITS"], errors='coerce')
                red_demerits = df[df["DEMERITS"] < 20]
                st.markdown(f"**Cadets with <20 Demerits ({len(red_demerits)})**")
                st.dataframe(red_demerits[["NAME", "DEMERITS"]], hide_index=True, use_container_width=True)


# =================================================================
#  MAIN DASHBOARD (Your original code)
# =================================================================
if st.session_state.role != "admin" or admin_dashboard_selection == "Main Dashboard":
    # --- UI ---
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    initial_idx = ["", *classes.keys()].index(st.session_state.selected_class or "")
    selected = st.selectbox("Select Class Level", ["", *classes.keys()], index=initial_idx)
    if selected != st.session_state.selected_class:
        st.session_state.update({"mode": "class", "selected_class": selected, "selected_cadet_display_name": None, "selected_cadet_cleaned_name": None})
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- CLASS VIEW ---
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

                    prev_df = sheet_df(acad_sheet_map[cls][term])
                    curr_df = sheet_df(acad_hist_map[cls][term])

                    prev_df.columns = [str(c).strip().upper() for c in prev_df.columns]
                    curr_df.columns = [str(c).strip().upper() for c in curr_df.columns]

                    prev_name_col = find_name_column(prev_df)
                    curr_name_col = find_name_column(curr_df)
                    
                    if prev_df.empty or prev_name_col is None:
                        st.warning("⚠️ No valid previous academic data or name column found.")
                    else:
                        prev_df["NAME_CLEANED"] = prev_df[prev_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                        row_prev = prev_df[prev_df["NAME_CLEANED"] == name_clean]

                        if row_prev.empty:
                            st.warning(f"No academic record found in previous sheet for {name_disp}.")
                            st.info("Some available cadet names: " + ", ".join(prev_df[prev_name_col].dropna().astype(str).unique()[:5]))
                        else:
                            row_prev = row_prev.iloc[0].drop([prev_name_col, "NAME_CLEANED"], errors='ignore')
                            
                            subjects = [s for s in row_prev.index.tolist() if s.upper() not in [c.upper() for c in cols_to_remove]]

                            df = pd.DataFrame({"SUBJECT": subjects})
                            
                            df["CURRENT GRADE"] = None
                            df["DEF/PROF POINTS"] = None
                            
                            if curr_name_col and not curr_df.empty:
                                curr_df["NAME_CLEANED"] = curr_df[curr_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                                row_curr = curr_df[curr_df["NAME_CLEANED"] == name_clean]
                                if not row_curr.empty:
                                    row_curr = row_curr.iloc[0]
                                    df["CURRENT GRADE"] = [pd.to_numeric(row_curr.get(subj, None), errors="coerce") for subj in subjects]
                                    df["DEF/PROF POINTS"] = [pd.to_numeric(row_curr.get("DEF/PROF POINTS", None), errors="coerce")] * len(subjects)
                            
                            df["STATUS"] = df["CURRENT GRADE"].apply(
                                lambda x: "PROFICIENT" if pd.notna(x) and x >= 7 else ("DEFICIENT" if pd.notna(x) else "")
                            )
                            
                            st.subheader("📝 Editable Grades and Points Table")
                            edited_df = st.data_editor(
                                df,
                                column_config={
                                    "SUBJECT": st.column_config.Column("SUBJECT", disabled=True),
                                    "CURRENT GRADE": st.column_config.NumberColumn("CURRENT GRADE", format="%f", step=0.1),
                                    "STATUS": st.column_config.Column("STATUS", disabled=True),
                                    "DEF/PROF POINTS": st.column_config.NumberColumn("DEF/PROF POINTS", format="%d", step=1),
                                },
                                hide_index=True,
                                use_container_width=True
                            )

                            grades_changed = not edited_df.equals(df)
                            
                            if grades_changed or st.session_state.get("force_show_submit", False):
                                st.success("✅ Detected changes. Click below to apply updates.")
                                if st.button("📤 Submit All Changes"):
                                    st.session_state["force_show_submit"] = False
                                    try:
                                        hist_ws = get_worksheet_by_name(acad_hist_map[cls][term])
                                        prev_ws = get_worksheet_by_name(acad_sheet_map[cls][term])
                                        
                                        hist_data = hist_ws.get_all_values()
                                        prev_data = prev_ws.get_all_values()
                                        
                                        headers_hist = hist_data[0]
                                        headers_prev = prev_data[0]
                                        
                                        name_idx_hist = next((i for i, h in enumerate(headers_hist) if h.upper() in [c.upper() for c in possible_name_cols]), None)
                                        name_idx_prev = next((i for i, h in enumerate(headers_prev) if h.upper() in [c.upper() for c in possible_name_cols]), None)
                                        
                                        if name_idx_hist is None or name_idx_prev is None:
                                            st.error("❌ 'NAME' column not found in one of the sheets.")
                                        else:
                                            hist_data, headers_hist = update_sheet_rows(hist_data, headers_hist, name_idx_hist, edited_df, name_clean, name_disp, "CURRENT GRADE")
                                            prev_data, headers_prev = update_sheet_rows(prev_data, headers_prev, name_idx_prev, edited_df, name_clean, name_disp, "PREVIOUS GRADE")
                                            
                                            hist_ws.clear()
                                            hist_ws.update(f"A1:{chr(64 + len(headers_hist))}{len(hist_data)}", [headers_hist] + hist_data[1:])
                                            prev_ws.clear()
                                            prev_ws.update(f"A1:{chr(64 + len(headers_prev))}{len(prev_data)}", [headers_prev] + prev_data[1:])

                                            st.cache_data.clear()
                                            st.success("✅ All changes saved to both sheets.")
                                    except Exception as e:
                                        st.error(f"❌ Error saving changes: {e}")
                            else:
                                st.session_state["force_show_submit"] = True
                                st.info("📝 No detected grade changes yet. Try editing a cell.")
                except Exception as e:
                    st.error(f"❌ Unexpected academic error: {e}")
                    
            with t3:
                try:
                    pft_sheet_map = {
                        "1CL": "1CL PFT",
                        "2CL": "2CL PFT",
                        "3CL": "3CL PFT"
                    }
                    
                    pft2_sheet_map = {
                        "1CL": "1CL PFT 2",
                        "2CL": "2CL PFT 2",
                        "3CL": "3CL PFT 2"
                    }
                    
                    term = st.selectbox("Select Term", ["1st Term", "2nd Term"])
                    
                    if 'cls' not in globals() or 'name_clean' not in globals() or 'name_disp' not in globals():
                        st.error("❌ Required context variables (cls, name_clean, name_disp) are not defined.")
                    else:
                    
                        def get_pft_data(sheet_key):
                            sheet_name = sheet_key.get(cls, None)
                            if not sheet_name:
                                return None, None, f"No PFT sheet mapped for selected class in {sheet_key}."
                            df = sheet_df(sheet_name)
                            if df is None or not isinstance(df, pd.DataFrame):
                                return None, None, f"❌ Sheet '{sheet_name}' did not return a valid DataFrame."
                            if df.empty:
                                return None, None, f"⚠️ No PFT data available in '{sheet_name}'."
                            df.columns = [c.strip().upper() for c in df.columns]
                            df["NAME_CLEANED"] = df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                            cadet = df[df["NAME_CLEANED"] == name_clean]
                            if cadet.empty:
                                return None, None, f"No PFT record found for {name_disp} in '{sheet_name}'."
                            return cadet.copy(), df, None
                    
                        exercises = [
                            ("Pushups", "PUSHUPS", "PUSHUPS_GRADES"),
                            ("Situps", "SITUPS", "SITUPS_GRADES"),
                            ("Pullups/Flexarm", "PULLUPS/FLEXARM", "PULLUPS_GRADES"),
                            ("3.2KM Run", "RUN", "RUN_GRADES")
                        ]
                    
                        def build_display_and_form(title, cadet_data, full_df, sheet_name):
                            updated_df = sheet_df(sheet_name)
                            updated_df.columns = [c.strip().upper() for c in updated_df.columns]
                            updated_df["NAME_CLEANED"] = updated_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                            cadet_data = updated_df[updated_df["NAME_CLEANED"] == name_clean].iloc[0]
                            full_df = updated_df.copy()
                    
                            st.subheader(title)
                            table = []
                            for label, raw_col, grade_col in exercises:
                                reps = cadet_data.get(raw_col, "")
                                grade = cadet_data.get(grade_col, "")
                                status = (
                                    "Passed" if str(grade).strip().replace('.', '', 1).isdigit() and float(grade) >= 7 else
                                    "Failed" if str(grade).strip().replace('.', '', 1).isdigit() else
                                    "N/A"
                                )
                                table.append({
                                    "Exercise": label,
                                    "Repetitions": reps,
                                    "Grade": grade,
                                    "Status": status
                                })
                            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
                    
                            with st.expander("✏️ Edit Form"):
                                cols = st.columns(2)
                                input_values = {}
                                for idx, (label, raw_col, grade_col) in enumerate(exercises):
                                    with cols[idx % 2]:
                                        reps = cadet_data.get(raw_col, "")
                                        grade = cadet_data.get(grade_col, "")
                                        input_values[raw_col] = st.number_input(
                                            f"{label} Reps",
                                            value=float(reps) if str(reps).replace('.', '', 1).isdigit() else 0.0,
                                            step=1.0, format="%g",
                                            key=f"{title}_{raw_col}"
                                        )
                                        input_values[grade_col] = st.number_input(
                                            f"{label} Grade",
                                            value=float(grade) if str(grade).replace('.', '', 1).isdigit() else 0.0,
                                            step=0.1, format="%g",
                                            key=f"{title}_{grade_col}"
                                        )
                    
                                if st.button(f"📂 Submit {title}"):
                                    for raw_col, val in input_values.items():
                                        full_df.loc[full_df["NAME_CLEANED"] == name_clean, raw_col] = val
                                    update_sheet(sheet_name, full_df)
                    
                                    sheet_df.clear()
                    
                                    st.success(f"✅ Changes to '{title}' saved successfully.")
                                    st.session_state["pft_refresh_triggered"] = True
                                    st.session_state["active_tab"] = "t3"
                                    time.sleep(1)
                                    st.rerun()
                    
                        if term == "1st Term":
                            cadet1, df1, err1 = get_pft_data(pft_sheet_map)
                            cadet2, df2, err2 = get_pft_data(pft2_sheet_map)
                            if err1:
                                st.warning(err1)
                            else:
                                build_display_and_form("🏋️ PFT 1 | 1st Term", cadet1.iloc[0], df1, pft_sheet_map[cls])
                            if err2:
                                st.warning(err2)
                            else:
                                build_display_and_form("🏋️ PFT 2 | 1st Term", cadet2.iloc[0], df2, pft2_sheet_map[cls])
                    
                        elif term == "2nd Term":
                            cadet2, df2, err2 = get_pft_data(pft2_sheet_map)
                            cadet1, df1, err1 = get_pft_data(pft_sheet_map)
                            if err2:
                                st.warning(err2)
                            else:
                                build_display_and_form("🏋️ PFT 2 | 2nd Term", cadet2.iloc[0], df2, pft2_sheet_map[cls])
                            if err1:
                                st.warning(err1)
                            else:
                                build_display_and_form("🏋️ PFT 1 | 2nd Term", cadet1.iloc[0], df1, pft_sheet_map[cls])
                    
                except Exception as e:
                    st.error(f"PFT load error: {e}")

            with t4:
                try:
                    mil_sheet_map = {
                        "1CL": "1CL MIL",
                        "2CL": "2CL MIL",
                        "3CL": "3CL MIL"
                    }
                    mil2_sheet_map = {
                        "1CL": "1CL MIL 2",
                        "2CL": "2CL MIL 2",
                        "3CL": "3CL MIL 2"
                    }
                    if 'cls' not in globals() or 'name_clean' not in globals() or 'name_disp' not in globals():
                        st.error("❌ Required context variables (cls, name_clean, name_disp) are not defined.")
                    else:
                        st.write("Military tab content here")
                except Exception as e:
                    st.error(f"❌ Military tab error: {e}")
