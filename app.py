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

if st.session_state.get("pft_refresh_triggered"):
    del st.session_state["pft_refresh_triggered"]

# Set default active tab for the Cadet Dashboard
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "t1"

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

# ✅ Create credentials object from secrets
credentials = Credentials.from_service_account_info(
    st.secrets["google_service_account"],
    scopes=scopes
)

# ✅ Authorize gspread with the credentials instance
gc = gspread.authorize(credentials)

# ✅ Open your spreadsheet
sh = gc.open("FOXTROT DASHBOARD V2")  # Replace with actual name

# ✅ Paste your update_sheet function here
def update_sheet(sheet_name, updated_df):
    try:
        worksheet = sh.worksheet(sheet_name)
        worksheet.clear()
        worksheet.update([updated_df.columns.values.tolist()] + updated_df.values.tolist())
    except Exception as e:
        st.error(f"❌ Failed to update Google Sheet '{sheet_name}': {e}")
        
if "last_report_fetch" not in st.session_state:
    st.session_state["last_report_fetch"] = 0
# --- Session State Initialization ---
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

# --- Logged In ---
st.sidebar.success(f"Logged in as **{st.session_state.username.upper()}** ({st.session_state.role})")

# Optional logout
if st.sidebar.button("🔓 Logout"):
    st.session_state.auth_ok = False
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()
# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Foxtrot CIS Dashboard",
    page_icon="🦊",  # Fox emoji icon
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
# -------------------- GOOGLE SHEETS --------------------
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

# -------------------- DEMOGRAPHICS --------------------
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
for cls_key, (start, end) in classes.items():
    demo_df.loc[start-1:end-1, "CLASS"] = cls_key

# -------------------- SESSION STATE --------------------
for key in ["mode", "selected_class", "selected_cadet_display_name", "selected_cadet_cleaned_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "mode" else "class"

# -------------------- SIDEBAR NAVIGATION --------------------
available_views = ["Cadet Dashboard"]
if st.session_state.role == "admin":
    available_views.append("Summary Dashboard")

selected_view = st.sidebar.selectbox("Select View", available_views)

# -------------------- MAIN DASHBOARD LOGIC --------------------
if selected_view == "Cadet Dashboard":
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
            
            # Tab for individual cadet details
            t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

            with t1:
                pic, info = st.columns([1, 2])
                with pic:
                    img_path = f"profile_pics/{name_disp}.jpg"
                    st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
                with info:
                    left, right = st.columns(2)
                    for idx, (k, v) in enumerate({k: v for k, v in row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]}.items()):
                        (left if idx % 2 == 0 else right).write(f"**{k}:** {v}")

            # --- [ACADEMICS TAB - Your existing code] ---
            with t2:
                # ... existing academics code here ...
                pass

            # --- [PFT TAB - Your existing code] ---
            with t3:
                # ... existing PFT code here ...
                pass

            # --- [MILITARY TAB - Your existing code] ---
            with t4:
                # ... existing military code here ...
                pass
            
            # --- [CONDUCT TAB - Your existing code] ---
            with t5:
                # ... existing conduct code here ...
                pass

# -------------------- SUMMARY DASHBOARD LOGIC (ADMIN ONLY) --------------------
elif selected_view == "Summary Dashboard":
    st.header("📊 Admin Summary Dashboard")
    st.info("This dashboard provides an overview of all cadets' performance.")
    
    # --- Section 1: Overall Performance Summary ---
    st.subheader("1. Overall Performance by Class")
    
    # Placeholder logic: You'll need to fetch data from all relevant sheets here
    summary_data = {
        "Class": ["1CL", "2CL", "3CL"],
        "Academics Deficient (%)": [20, 15, 10],  # Example data
        "PFT Deficient (%)": [5, 8, 3],          # Example data
        "Military Deficient (%)": [10, 5, 2],    # Example data
        "Conduct Violations (Avg)": [2.5, 1.8, 0.5] # Example data
    }
    
    summary_df = pd.DataFrame(summary_data).set_index("Class")
    st.dataframe(summary_df)

    st.markdown("---")

    # --- Section 2: Deficient Cadets Report ---
    st.subheader("2. Deficient Cadets Report")
    
    # This is the crucial part. You need to read all data and consolidate it.
    # Here's a conceptual outline.
    
    deficient_cadets_list = []
    
    # Using the demographics data as the master list of all cadets
    for _, cadet_row in demo_df.iterrows():
        name_display = cadet_row.get("FULL NAME_DISPLAY")
        name_clean = cadet_row.get("FULL NAME")
        cls = cadet_row.get("CLASS")

        if not cls: continue

        deficiencies = []

        # --- Check for academic deficiency ---
        # Looping through both terms to be thorough
        try:
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

            for term_key in ["1st Term", "2nd Term"]:
                acad_hist_sheet_name = acad_hist_map[cls][term_key]
                df_acad = sheet_df(acad_hist_sheet_name)
                if not df_acad.empty:
                    name_col = next((c for c in df_acad.columns if 'NAME' in c.upper()), None)
                    if name_col:
                        df_acad["NAME_CLEANED"] = df_acad[name_col].astype(str).apply(clean_cadet_name_for_comparison)
                        cadet_acad_data = df_acad[df_acad["NAME_CLEANED"] == name_clean]
                        if not cadet_acad_data.empty:
                            grade_cols = [col for col in cadet_acad_data.columns if col not in [name_col, "NAME_CLEANED", "CLASS", "TERM"]]
                            average_grade = pd.to_numeric(cadet_acad_data[grade_cols].iloc[0], errors='coerce').mean()
                            if average_grade < 7:
                                deficiencies.append(f"Academics ({term_key})")
                                # No break here, as we want to find all deficiencies
        except Exception:
            pass
        
        # --- Check for PFT deficiency ---
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

            for pft_sheet in [pft_sheet_map.get(cls), pft2_sheet_map.get(cls)]:
                if pft_sheet:
                    df_pft = sheet_df(pft_sheet)
                    if not df_pft.empty:
                        df_pft["NAME_CLEANED"] = df_pft.get('NAME', pd.Series(dtype='object')).astype(str).apply(clean_cadet_name_for_comparison)
                        cadet_pft_data = df_pft[df_pft["NAME_CLEANED"] == name_clean]
                        if not cadet_pft_data.empty:
                            pft_grades = ['PUSHUPS_GRADES', 'SITUPS_GRADES', 'PULLUPS_GRADES', 'RUN_GRADES']
                            if any(pd.to_numeric(cadet_pft_data.iloc[0].get(g), errors='coerce') < 7 for g in pft_grades if g in cadet_pft_data.columns):
                                deficiencies.append(f"PFT")
                                break  # One PFT deficiency is enough to report
        except Exception:
            pass

        # --- Check for military deficiency ---
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

            for mil_sheet in [mil_sheet_map.get(cls), mil2_sheet_map.get(cls)]:
                if mil_sheet:
                    df_mil = sheet_df(mil_sheet)
                    if not df_mil.empty:
                        df_mil["NAME_CLEANED"] = df_mil.get('NAME', pd.Series(dtype='object')).astype(str).apply(clean_cadet_name_for_comparison)
                        cadet_mil_data = df_mil[df_mil["NAME_CLEANED"] == name_clean]
                        if not cadet_mil_data.empty:
                            if cls == "1CL" and 'GRADE' in cadet_mil_data.columns and pd.to_numeric(cadet_mil_data.iloc[0].get('GRADE'), errors='coerce') < 7:
                                deficiencies.append("Military")
                            elif cls == "2CL":
                                mil_grades = ['AS', 'NS', 'AFS']
                                if any(pd.to_numeric(cadet_mil_data.iloc[0].get(g), errors='coerce') < 7 for g in mil_grades if g in cadet_mil_data.columns):
                                    deficiencies.append("Military")
                            elif cls == "3CL" and 'MS231' in cadet_mil_data.columns and pd.to_numeric(cadet_mil_data.iloc[0].get('MS231'), errors='coerce') < 7:
                                deficiencies.append("Military")
        except Exception:
            pass
        
        # --- Add other checks for conduct, etc. here ---
        
        if deficiencies:
            unique_deficiencies = sorted(list(set(deficiencies)))
            deficient_cadets_list.append({
                "Cadet Name": name_display,
                "Class": cls,
                "Deficiencies": ", ".join(unique_deficiencies)
            })

    if deficient_cadets_list:
        deficient_df = pd.DataFrame(deficient_cadets_list)
        st.dataframe(deficient_df, use_container_width=True)
    else:
        st.success("No deficient cadets found! All cadets are performing at or above standard.")
