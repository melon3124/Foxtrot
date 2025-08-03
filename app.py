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
# --- SUMMARY DASHBOARD FIXED VERSION ---
if st.session_state.view == "summary":
    st.set_page_config(layout="wide")
    st.title("📊 Summary Dashboard")

    term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_term")

    acad_hist_map = {
        "1CL": {"1st Term": "1CL ACAD HISTORY", "2nd Term": "1CL ACAD HISTORY 2"},
        "2CL": {"1st Term": "2CL ACAD HISTORY", "2nd Term": "2CL ACAD HISTORY 2"},
        "3CL": {"1st Term": "3CL ACAD HISTORY", "2nd Term": "3CL ACAD HISTORY 2"}
    }

    pft_sheet_map = {
        "1CL": {"1st Term": "1CL PFT", "2nd Term": "1CL PFT 2"},
        "2CL": {"1st Term": "2CL PFT", "2nd Term": "2CL PFT 2"},
        "3CL": {"1st Term": "3CL PFT", "2nd Term": "3CL PFT 2"}
    }

    mil_sheet_map = {
        "1CL": {"1st Term": "1CL MIL", "2nd Term": "1CL MIL 2"},
        "2CL": {"1st Term": "2CL MIL", "2nd Term": "2CL MIL 2"},
        "3CL": {"1st Term": "3CL MIL", "2nd Term": "3CL MIL 2"}
    }

    conduct_sheet_map = {
        "1CL": {"1st Term": "1CL CONDUCT", "2nd Term": "1CL CONDUCT 2"},
        "2CL": {"1st Term": "2CL CONDUCT", "2nd Term": "2CL CONDUCT 2"},
        "3CL": {"1st Term": "3CL CONDUCT", "2nd Term": "3CL CONDUCT 2"}
    }

    demo_df = sheet_df("DEMOGRAPHICS")
    demo_df.columns = [c.strip().upper().replace("\xa0", "") for c in demo_df.columns]
    demo_df["FULL NAME"] = demo_df.apply(
        lambda r: clean_cadet_name_for_comparison(
            f"{r.get('FAMILY NAME','').strip()}, {r.get('FIRST NAME','').strip()} {r.get('MIDDLE NAME','').strip()} {r.get('EXTN','').strip()}"
        ), axis=1
    )
    demo_df["FULL NAME_DISPLAY"] = demo_df.apply(
        lambda r: f"{r.get('FAMILY NAME','').strip()}, {r.get('FIRST NAME','').strip()} {r.get('MIDDLE NAME','').strip()} {r.get('EXTN','').strip()}".strip(), axis=1
    )

    acad_tab, pft_tab, mil_tab, conduct_tab = st.tabs(["📚 Academics", "🏃 PFT", "🫦 Military", "⚖ Conduct"])

    with acad_tab:
        st.subheader("📚 Academic Summary")
        for cls in classes:
            sheet_name = acad_hist_map[cls][term]
            acad_df = sheet_df(sheet_name)
            if acad_df.empty:
                continue

            st.markdown(f"### {cls} Academic Performance")
            subject_cols = [col for col in acad_df.columns if col not in ["NAME", "NAME_CLEANED"]]

            for subject in subject_cols:
                acad_df[subject] = pd.to_numeric(acad_df[subject], errors='coerce')
                prof = acad_df[acad_df[subject] >= 7][["NAME", subject]].dropna().sort_values(by=subject, ascending=False)
                defn = acad_df[acad_df[subject] < 7][["NAME", subject]].dropna().sort_values(by=subject)

                st.markdown(f"**Subject: {subject}**")
                st.write("Proficient Cadets")
                st.dataframe(prof, use_container_width=True)
                st.write("Deficient Cadets")
                st.dataframe(defn, use_container_width=True)

                if not defn.empty:
                    max_def = defn.sort_values(by=subject).head(1)
                    st.write("⬇️ Highest Deficiency")
                    st.dataframe(max_def, use_container_width=True)

    with pft_tab:
        st.subheader("🏃 PFT Summary")
        for cls in classes:
            sheet_name = pft_sheet_map[cls][term]
            pft_df = sheet_df(sheet_name)
            if pft_df.empty:
                continue

            st.markdown(f"### {cls} PFT Summary")
            for col in ["PUSHUPS_GRADES", "SITUPS_GRADES", "PULLUPS_GRADES", "RUN_GRADES"]:
                pft_df[col] = pd.to_numeric(pft_df[col], errors='coerce')
            pft_df["AVG_GRADE"] = pft_df[["PUSHUPS_GRADES", "SITUPS_GRADES", "PULLUPS_GRADES", "RUN_GRADES"]].mean(axis=1)
            pft_df["NAME_CLEANED"] = pft_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)

            pft_df["GENDER"] = pft_df["GENDER"].astype(str).str.upper().str.strip()

            smc = pft_df[pft_df["AVG_GRADE"] < 7]
            st.write("🚫 SMC Cadets (Failed)")
            st.dataframe(smc[["NAME", "AVG_GRADE"]], use_container_width=True)

            top_male = pft_df[pft_df["GENDER"] == "M"].sort_values("AVG_GRADE", ascending=False).head(1)
            top_female = pft_df[pft_df["GENDER"] == "F"].sort_values("AVG_GRADE", ascending=False).head(1)

            st.write("💪 Strongest Male Cadet")
            if not top_male.empty and all(col in top_male.columns for col in ["NAME", "AVG_GRADE"]):
                st.dataframe(top_male[["NAME", "AVG_GRADE"]], use_container_width=True)
            else:
                st.warning("No male cadet data available.")

            st.write("💪 Strongest Female Cadet")
            if not top_female.empty and all(col in top_female.columns for col in ["NAME", "AVG_GRADE"]):
                st.dataframe(top_female[["NAME", "AVG_GRADE"]], use_container_width=True)
            else:
                st.warning("No female cadet data available.")

    with mil_tab:
        st.subheader("🫦 Military Summary")
        for cls in classes:
            sheet_name = mil_sheet_map[cls][term]
            mil_df = sheet_df(sheet_name)
            if mil_df.empty:
                continue

            st.markdown(f"### {cls} Military Grades")
            if cls == "1CL":
                mil_df["GRADE"] = pd.to_numeric(mil_df["GRADE"], errors="coerce")
                prof = mil_df[mil_df["GRADE"] >= 7]
                defn = mil_df[mil_df["GRADE"] < 7]
            elif cls == "2CL":
                for col in ["AS", "NS", "AFS"]:
                    mil_df[col] = pd.to_numeric(mil_df[col], errors="coerce")
                prof = mil_df[(mil_df["AS"] >= 7) & (mil_df["NS"] >= 7) & (mil_df["AFS"] >= 7)]
                defn = mil_df[(mil_df["AS"] < 7) | (mil_df["NS"] < 7) | (mil_df["AFS"] < 7)]
            elif cls == "3CL":
                mil_df["MS231"] = pd.to_numeric(mil_df["MS231"], errors="coerce")
                prof = mil_df[mil_df["MS231"] >= 7]
                defn = mil_df[mil_df["MS231"] < 7]

            st.write("✅ Proficient Cadets")
            st.dataframe(prof[["NAME"]], use_container_width=True)
            st.write("🚫 Deficient Cadets")
            st.dataframe(defn[["NAME"]], use_container_width=True)

    with conduct_tab:
        st.subheader("⚖ Conduct Summary")
        for cls in classes:
            sheet_name = conduct_sheet_map[cls][term]
            conduct_df = sheet_df(sheet_name)
            if conduct_df.empty:
                continue

            st.markdown(f"### {cls} Conduct")
            if "MERITS" in conduct_df.columns:
                conduct_df["DEMERITS"] = pd.to_numeric(conduct_df["MERITS"], errors="coerce")
            else:
                st.warning("⚠️ 'MERITS' column not found in conduct data.")
                conduct_df["DEMERITS"] = 0

            touring_df = sheet_df("REPORTS")
            touring_df["NAME_CLEANED"] = touring_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
            if "CLASS" in conduct_df.columns:
                class_cadets = conduct_df[conduct_df["CLASS"] == cls]["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
            else:
                st.warning("⚠️ 'CLASS' column not found in conduct data.")
                class_cadets = pd.Series(dtype=str)
            touring_filtered = touring_df[touring_df["NAME_CLEANED"].isin(class_cadets)]

            st.write("🎒 Touring Cadets")
            st.dataframe(touring_filtered[["NAME", "DATE OF REPORT", "NATURE", "DEMERITS"]], use_container_width=True)

            flagged = conduct_df[conduct_df["DEMERITS"] < 20]
            st.write("🔴 Cadets with < 20 Demerits")
            st.dataframe(flagged[["NAME", "DEMERITS"]], use_container_width=True)

    st.stop()


    # -------------------- SESSION STATE --------------------
for key in ["mode", "selected_class", "selected_cadet_display_name", "selected_cadet_cleaned_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "mode" else "class"

# -------------------- UI --------------------
st.markdown('<div class="centered">', unsafe_allow_html=True)
initial_idx = ["", *classes.keys()].index(st.session_state.selected_class or "")
selected = st.selectbox("Select Class Level", ["", *classes.keys()], index=initial_idx)
if selected != st.session_state.selected_class:
    st.session_state.update({"mode": "class", "selected_class": selected, "selected_cadet_display_name": None, "selected_cadet_cleaned_name": None})
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- CLASS VIEW --------------------
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
        
# --- Your existing code with the improved t1 section integrated ---
    name_disp = st.session_state.selected_cadet_display_name
    name_clean = st.session_state.selected_cadet_cleaned_name
    if name_clean:
        row = demo_df[demo_df["FULL NAME"] == name_clean].iloc[0]
        st.markdown(f"## Showing details for: {name_disp}")
        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

        with t1:
            # Use a two-column layout for the profile picture and information
            pic, info = st.columns([1, 2])
            
            with pic:
                img_path = f"profile_pics/{name_disp}.jpg"
                # Display the profile image or a placeholder if not found
                st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
            
            with info:
                # Group the personal information inside a collapsible expander
                with st.expander("📝 Cadet Information", expanded=True):
                    
                    # Create a two-column grid for the key-value pairs
                    col1, col2 = st.columns(2)
                    
                    # Create a dictionary of the information to display, excluding internal keys
                    info_to_display = {
                        k: v for k, v in row.items() 
                        if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]
                    }
                    
                    # Iterate through the items and place them in alternating columns
                    for idx, (k, v) in enumerate(info_to_display.items()):
                        target_col = col1 if idx % 2 == 0 else col2
                        target_col.markdown(f"**{k}:** {v}")
                        
                # Add an informational note at the bottom
                st.info("💡 You can find more detailed academic, military, and PFT data in the other tabs.")

    with t2:
        try:
            if "selected_term" not in st.session_state:
                st.session_state.selected_term = "1st Term"
    
            term = st.radio(
                "Select Term",
                ["1st Term", "2nd Term"],
                index=["1st Term", "2nd Term"].index(st.session_state.selected_term),
                horizontal=True,
                help="Choose academic term for grade input"
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
    
            def update_sheet_rows(data, headers, name_idx, subj_idx_map, edited_df, name_clean, name_disp):
                updated = False
                for row in data[1:]:
                    if clean_cadet_name_for_comparison(row[name_idx]) == name_clean:
                        for _, r in edited_df.iterrows():
                            subj = r["SUBJECT"]
                            val = str(r["CURRENT GRADE"]) if pd.notna(r["CURRENT GRADE"]) else ""
                            row[subj_idx_map[subj]] = val
                        updated = True
                        break
                if not updated:
                    new_row = ["" for _ in headers]
                    new_row[name_idx] = name_disp
                    for _, r in edited_df.iterrows():
                        subj = r["SUBJECT"]
                        val = str(r["CURRENT GRADE"]) if pd.notna(r["CURRENT GRADE"]) else ""
                        new_row[subj_idx_map[subj]] = val
                    data.append(new_row)
                return data
    
            prev_df = sheet_df(acad_sheet_map[cls][term])
            prev_df.columns = [str(c).strip().upper() for c in prev_df.columns]
            prev_name_col = find_name_column(prev_df)
    
            if prev_df.empty or prev_name_col is None:
                st.warning("⚠️ No valid academic data or name column found.")
            else:
                prev_df["NAME_CLEANED"] = prev_df[prev_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                row_prev = prev_df[prev_df["NAME_CLEANED"] == name_clean]
    
                if row_prev.empty:
                    st.warning(f"No academic record found for {name_disp}.")
                    st.info("Some available cadet names: " + ", ".join(prev_df[prev_name_col].dropna().astype(str).unique()[:5]))
                else:
                    row_prev = row_prev.iloc[0].drop([prev_name_col, "NAME_CLEANED"], errors='ignore')
                    subjects = row_prev.index.tolist()
                    df = pd.DataFrame({"SUBJECT": subjects})
                    df["CURRENT GRADE"] = [pd.to_numeric(row_prev.get(subj, None), errors="coerce") for subj in subjects]
    
                    # ❌ Remove unwanted rows
                    df = df[~df["SUBJECT"].str.upper().isin(["PREVIOUS GRADE", "DEF/PROF POINTS"])]
    
                    # ✅ Add status label
                    def grade_status(g):
                        if pd.isna(g):
                            return "⚪️ NO GRADE"
                        elif g < 7.00:
                            return "🔴 DEFICIENT"
                        else:
                            return "🟢 PROFICIENT"
    
                    df["STATUS"] = df["CURRENT GRADE"].apply(grade_status)
    
                    # 📋 Display current grades with status
                    st.subheader("📋 Current Grades")
                    st.dataframe(df[["SUBJECT", "CURRENT GRADE", "STATUS"]], use_container_width=True, hide_index=True)
    
                    # 📝 Edit grades with text input (not dropdown)
                    st.subheader("📝 Edit Grades (Type Grades Directly)")
    
                    df_editor = df.copy()
                    df_editor["GRADE INPUT"] = df_editor["CURRENT GRADE"].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else ""
                    )
    
                    edited_df = st.data_editor(
                        df_editor[["SUBJECT", "GRADE INPUT"]],
                        column_config={
                            "GRADE INPUT": st.column_config.TextColumn("Grade")
                        },
                        use_container_width=True,
                        hide_index=True,
                        key="grade_edit_table"
                    )
    
                    # Try converting typed grades to float
                    def try_float(x):
                        try:
                            return float(x)
                        except:
                            return None
    
                    df["UPDATED GRADE"] = edited_df["GRADE INPUT"].apply(try_float)
                    grades_changed = not df["CURRENT GRADE"].equals(df["UPDATED GRADE"])
    
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
                                    subj_idx_hist = {subj: headers_hist.index(subj) if subj in headers_hist else headers_hist.append(subj) or len(headers_hist) - 1 for subj in df["SUBJECT"]}
                                    subj_idx_prev = {subj: headers_prev.index(subj) if subj in headers_prev else headers_prev.append(subj) or len(headers_prev) - 1 for subj in df["SUBJECT"]}
    
                                    for row in hist_data[1:]: row.extend([""] * (len(headers_hist) - len(row)))
                                    for row in prev_data[1:]: row.extend([""] * (len(headers_prev) - len(row)))
    
                                    edit_df_for_save = pd.DataFrame({
                                        "SUBJECT": df["SUBJECT"],
                                        "CURRENT GRADE": df["UPDATED GRADE"]
                                    })
    
                                    hist_data = update_sheet_rows(hist_data, headers_hist, name_idx_hist, subj_idx_hist, edit_df_for_save, name_clean, name_disp)
                                    prev_data = update_sheet_rows(prev_data, headers_prev, name_idx_prev, subj_idx_prev, edit_df_for_save, name_clean, name_disp)
    
                                    hist_ws.clear()
                                    hist_ws.update("A1", [headers_hist] + hist_data[1:])
                                    prev_ws.clear()
                                    prev_ws.update("A1", [headers_prev] + prev_data[1:])
    
                                    st.cache_data.clear()
                                    st.success("✅ All changes saved to both sheets.")
                            except Exception as e:
                                st.error(f"❌ Error saving changes: {e}")
                    else:
                        st.session_state["force_show_submit"] = True
                        st.info("📝 No detected grade changes yet. Try editing a value.")
    
        except Exception as e:
            st.error(f"❌ Unexpected academic error: {e}")


        with t3:
            st.markdown("### 🏃‍♂️ PFT Scores")
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
                        # Re-fetch latest data after reload
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
                                "✅ Passed" if str(grade).strip().replace('.', '', 1).isdigit() and float(grade) >= 7 else
                                "🚫 Failed" if str(grade).strip().replace('.', '', 1).isdigit() else
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
            
                                # 🔻 Clear cache so updated data is fetched
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
            st.markdown("### 🎖️ Military Grades")
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
                    term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="mil_term")
            
                    sheet_name = mil_sheet_map.get(cls) if term == "1st Term" else mil2_sheet_map.get(cls)
                    if not sheet_name:
                        st.warning(f"No sheet mapped for class {cls} in {term}.")
                    else:
                        df = sheet_df(sheet_name)
                        if df is None or df.empty:
                            st.info(f"No military data found in '{sheet_name}'.")
                        else:
                            df.columns = [c.strip().upper() for c in df.columns]
                            df["NAME_CLEANED"] = df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                            cadet_df = df[df["NAME_CLEANED"] == name_clean].copy()
            
                            if cadet_df.empty:
                                st.warning(f"No military record found for {name_disp} in '{sheet_name}'.")
                            else:
                                st.subheader(f"📋 Military Grade Summary – {term}")
                                display_rows = []
            
                                if cls == "1CL":
                                    grade = cadet_df.iloc[0].get("GRADE", "N/A")
                                    try:
                                        status = "✅ Proficient" if float(grade) >= 7 else "🚫 DEFICIENT"
                                    except:
                                        status = "N/A"
                                    display_rows.append({
                                        "Name": name_disp,
                                        "BOS": cadet_df.iloc[0].get("BOS", ""),
                                        "GRADE": grade,
                                        "Status": status
                                    })
            
                                elif cls == "2CL":
                                    for subj in ["AS", "NS", "AFS"]:
                                        grade = cadet_df.iloc[0].get(subj, "N/A")
                                        try:
                                            status = "✅ Proficient" if float(grade) >= 7 else "🚫 DEFICIENT"
                                        except:
                                            status = "N/A"
                                        display_rows.append({
                                            "Name": name_disp,
                                            "Subject": subj,
                                            "GRADE": grade,
                                            "Status": status
                                        })
            
                                elif cls == "3CL":
                                    grade = cadet_df.iloc[0].get("MS231", "N/A")
                                    try:
                                        status = "✅ Proficient" if float(grade) >= 7 else "🚫 DEFICIENT"
                                    except:
                                        status = "N/A"
                                    display_rows.append({
                                        "Name": name_disp,
                                        "MS231": grade,
                                        "Status": status
                                    })
            
                                st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
            
                                st.markdown(f"### ✏️ Edit Grades – {term}")
                                input_data = {}
            
                                if cls == "1CL":
                                    current_grade = cadet_df.iloc[0].get("GRADE", "")
                                    input_data["GRADE"] = st.number_input(f"{term} Grade", value=float(current_grade) if str(current_grade).replace('.', '', 1).isdigit() else 0.0, step=0.1)
            
                                elif cls == "2CL":
                                    for subj in ["AS", "NS", "AFS"]:
                                        current_grade = cadet_df.iloc[0].get(subj, "")
                                        input_data[subj] = st.number_input(
                                            f"{subj} Grade – {term}", value=float(current_grade) if str(current_grade).replace('.', '', 1).isdigit() else 0.0, step=0.1
                                        )
            
                                elif cls == "3CL":
                                    current_grade = cadet_df.iloc[0].get("MS231", "")
                                    input_data["MS231"] = st.number_input(
                                        f"MS231 Grade – {term}", value=float(current_grade) if str(current_grade).replace('.', '', 1).isdigit() else 0.0, step=0.1
                                    )
            
                                if st.button(f"📂 Submit Changes – {term}"):
                                    full_df = sheet_df(sheet_name)
                                    full_df.columns = [c.strip().upper() for c in full_df.columns]
                                    full_df["NAME_CLEANED"] = full_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
            
                                    for col, val in input_data.items():
                                        full_df.loc[full_df["NAME_CLEANED"] == name_clean, col] = val
            
                                    full_df.drop(columns=["NAME_CLEANED"], inplace=True)
                                    update_sheet(sheet_name, full_df)
                                    sheet_df.clear()
                                    st.success(f"✅ {term} military grades updated successfully.")
                                    st.rerun()
            
            except Exception as e:
                st.error(f"Military tab error: {e}")
        
        with t5:
            st.markdown("### 📄 Conduct Reports")
            try:
                # Sheet map per term
                conduct_sheet_map = {
                    "1st Term": {
                        "1CL": "1CL CONDUCT",
                        "2CL": "2CL CONDUCT",
                        "3CL": "3CL CONDUCT"
                    },
                    "2nd Term": {
                        "1CL": "1CL CONDUCT 2",
                        "2CL": "2CL CONDUCT 2",
                        "3CL": "3CL CONDUCT 2"
                    }
                }
            
                term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="conduct_term")
                sheet_name = conduct_sheet_map[term].get(cls)
            
                if not sheet_name:
                    st.warning("Please select a valid class to view conduct data.")
                else:
                    conduct = sheet_df(sheet_name)
                    conduct.columns = [c.strip().lower() for c in conduct.columns]
                    conduct["name_cleaned"] = conduct["name"].astype(str).apply(clean_cadet_name_for_comparison)
                    cadet_data = conduct[conduct["name_cleaned"] == name_clean].copy()
            
                    if cadet_data.empty:
                        st.warning(f"No conduct data found for {name_disp} in {sheet_name}.")
                    else:
                        # --- Merits Summary + Edit ---
                        st.subheader("📜 Merits Summary")
            
                        current_merits = cadet_data.iloc[0].get("merits", "0")
                        merits_value = st.number_input(
                            f"Edit Merits – {term}",
                            value=float(current_merits) if str(current_merits).replace('.', '', 1).lstrip('-').isdigit() else 0.0,
                            step=1.0
                        )
            
                        status = "🚫 Failed" if merits_value < 0 else "✅ Passed"
            
                        merit_table = pd.DataFrame([{
                            "Name": name_disp,
                            "Merits": merits_value,
                            "Status": status
                        }])
                        st.dataframe(merit_table, hide_index=True, use_container_width=True)
            
                        if st.button(f"💾 Save Merits – {term}"):
                            try:
                                full_df = sheet_df(sheet_name)
                                full_df.columns = [c.strip().lower() for c in full_df.columns]
                                full_df["name_cleaned"] = full_df["name"].astype(str).apply(clean_cadet_name_for_comparison)
                                full_df.loc[full_df["name_cleaned"] == name_clean, "merits"] = merits_value
                                full_df.drop(columns=["name_cleaned"], inplace=True)
                                update_sheet(sheet_name, full_df)
                                sheet_df.clear()
                                st.success("✅ Merits updated successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to update merits: {e}")
            
                        # --- Conduct Reports Table ---
                        st.subheader("📋 Conduct Reports")
                        expected_cols = ["NAME", "REPORT", "DATE OF REPORT", "NATURE", "DEMERITS"]
            
                        if "last_report_fetch" not in st.session_state:
                            st.session_state["last_report_fetch"] = 0
            
                        try:
                            now = time.time()
                            if now - st.session_state["last_report_fetch"] > 10:
                                reports_df = sheet_df("REPORTS")
                                st.session_state["last_report_df"] = reports_df
                                st.session_state["last_report_fetch"] = now
                            else:
                                reports_df = st.session_state.get("last_report_df", pd.DataFrame(columns=expected_cols))
            
                            reports_df.columns = [c.strip().upper() for c in reports_df.columns]
            
                            if not set(expected_cols).issubset(set(reports_df.columns)):
                                st.warning("⚠️ 'REPORTS' sheet is missing required columns. Showing empty table.")
                                cadet_reports = pd.DataFrame(columns=expected_cols)
                            else:
                                reports_df["NAME_CLEANED"] = reports_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                                cadet_reports = reports_df[reports_df["NAME_CLEANED"] == name_clean]
            
                        except Exception as e:
                            st.warning(f"⚠️ Could not load reports sheet: {e}")
                            cadet_reports = pd.DataFrame(columns=expected_cols)
            
                        st.dataframe(
                            cadet_reports[["NAME", "REPORT", "DATE OF REPORT", "NATURE", "DEMERITS"]],
                            use_container_width=True,
                            hide_index=True
                        )
            
                        # --- Add New Report Form ---
                        st.subheader("➕ Add New Conduct Report")
                        with st.form("report_form"):
                            new_report = st.text_area("Report Description", placeholder="Enter behavior details...")
                            new_report_date = st.date_input("Date of Report")
                            new_nature = st.selectbox("Nature", ["I", "II", "III", "IV"])
                            new_demerits = st.number_input("Demerits", step=1)
                            submitted = st.form_submit_button("📤 Submit Report")
            
                        if submitted:
                            try:
                                time.sleep(0.5)  # Allow smoother API write
                                report_ws = SS.worksheet("REPORTS")
                                new_row = [
                                    name_disp,
                                    new_report.strip(),
                                    str(new_report_date),
                                    new_nature,
                                    str(new_demerits)
                                ]
                                report_ws.append_row(new_row, value_input_option="USER_ENTERED")
                                st.cache_data.clear()
                                time.sleep(0.75)
                                st.success("✅ Report submitted successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error submitting to 'REPORTS' sheet: {e}")
            
            except Exception as e:
                st.error(f"❌ Unexpected error in Conduct tab: {e}")
