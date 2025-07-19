import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import re
import unicodedata
st.write("Version:", st.__version__)
# -------------------- SIMPLE AUTH --------------------
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
            "password": "CadetV13wer",
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
for cls, (start, end) in classes.items():
    demo_df.iloc[start-1:end, demo_df.columns.get_loc("CLASS")] = cls

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

    name_disp = st.session_state.selected_cadet_display_name
    name_clean = st.session_state.selected_cadet_cleaned_name
    if name_clean:
        row = demo_df[demo_df["FULL NAME"] == name_clean].iloc[0]
        st.markdown(f"## Showing details for: {name_disp}")
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

        with t2:  # Academics tab
            try:
                acad_sheet_map = {
                    "1CL": "1CL ACAD",
                    "2CL": "2CL ACAD",
                    "3CL": "3CL ACAD"
                }
                acad = sheet_df(acad_sheet_map[cls])

                if acad.empty:
                    st.info("No Academic data available for this class.")
                else:
                    target_name_col = "NAME"
                    if target_name_col not in acad.columns:
                        st.error(f"Error: Expected column '{target_name_col}' not found in the academic sheet '{acad_sheet_map[cls]}'.")
                        st.write(f"Available columns in '{acad_sheet_map[cls]}': {acad.columns.tolist()}")
                    else:
                        acad['NAME_CLEANED'] = acad[target_name_col].astype(str).apply(clean_cadet_name_for_comparison)

                        r = acad[acad["NAME_CLEANED"] == name_clean]

                        if not r.empty:
                            r = r.iloc[0]
                            df_data = r.drop([col for col in r.index if col in [target_name_col, 'NAME_CLEANED']], errors='ignore')

                            df = pd.DataFrame({"Subject": df_data.index, "Grade": df_data.values})
                            df["Grade_Numeric"] = pd.to_numeric(df["Grade"], errors='coerce')
                            df["Status"] = df["Grade_Numeric"].apply(lambda g: "Proficient" if g >= 7 else "Deficient" if pd.notna(g) else "N/A")

                            st.dataframe(df[['Subject', 'Grade', 'Status']], hide_index=True)
                        else:
                            st.warning(f"No academic record found for {name_disp}.")
            except Exception as e:
                st.error(f"Academic load error: {e}")

        with t3:
            try:
                pft_sheet_map = {
                    "1CL": "1CL PFT",
                    "2CL": "2CL PFT",
                    "3CL": "3CL PFT"
                }
        
                sheet_name = pft_sheet_map.get(cls, None)
                if not sheet_name:
                    st.warning("No PFT sheet mapped for selected class.")
                else:
                    pft = sheet_df(sheet_name)
        
                    if pft.empty:
                        st.info(f"No PFT data available in '{sheet_name}'.")
                    else:
                        # Normalize columns (strip spaces)
                        pft.columns = [c.strip().upper() for c in pft.columns]
                        pft["NAME_CLEANED"] = pft["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
        
                        cadet = pft[pft["NAME_CLEANED"] == name_clean]
        
                        if cadet.empty:
                            st.warning(f"No PFT record found for {name_disp} in '{sheet_name}'.")
                        else:
                            cadet = cadet.iloc[0]
        
                            exercises = [
                                ("Pushups", "PUSHUPS", "PUSHUPS_GRADES"),
                                ("Situps", "SITUPS", "SITUPS_GRADES"),
                                ("Pullups/Flexarm", "PULLUPS/FLEXARM", "PULLUPS_GRADES"),
                                ("3.2KM Run", "RUN", "RUN_GRADES")
                            ]
        
                            table = []
                            for label, raw_col, grade_col in exercises:
                                reps = cadet.get(raw_col, "")
                                grade = cadet.get(grade_col, "N/A")
                                status = (
                                    "Passed" if str(grade).strip().isdigit() and int(grade) >= 3
                                    else "Failed" if str(grade).strip().isdigit()
                                    else "N/A"
                                )
                                table.append({
                                    "Exercise": label,
                                    "Repetitions": reps,
                                    "Grade": grade,
                                    "Status": status
                                })
        
                            st.dataframe(pd.DataFrame(table), hide_index=True)
            except Exception as e:
                st.error(f"PFT load error: {e}")
        with t4:
            try:
                mil_sheet_map = {
                    "1CL": "1CL MIL",
                    "2CL": "2CL MIL",
                    "3CL": "3CL MIL"
                }
                sheet_name = mil_sheet_map.get(cls)
                if not sheet_name:
                    st.warning("Select a class to view military grades.")
                else:
                    mil = sheet_df(sheet_name)
                    if mil.empty:
                        st.info(f"No military data found in '{sheet_name}'.")
                    else:
                        mil.columns = [c.strip().upper() for c in mil.columns]
        
                        name_col = "NAME"
                        if name_col not in mil.columns:
                            st.error(f"Expected 'NAME' column not found in '{sheet_name}'. Got: {mil.columns.tolist()}")
                        else:
                            mil["NAME_CLEANED"] = mil[name_col].astype(str).apply(clean_cadet_name_for_comparison)
                            cadet = mil[mil["NAME_CLEANED"] == name_clean]
        
                            if cadet.empty:
                                st.warning(f"No military record found for {name_disp} in '{sheet_name}'.")
                            else:
                                cadet = cadet.iloc[0]
        
                                if cls == "1CL":
                                    bos = cadet.get("BOS", "N/A")
                                    grade = cadet.get("GRADE", "N/A")
                                    try:
                                        grade_val = float(grade)
                                        status = "Proficient" if grade_val >= 7 else "DEFICIENT"
                                    except:
                                        status = "N/A"
                                    df = pd.DataFrame([{
                                        "Name": name_disp,
                                        "BOS": bos,
                                        "Grade": grade,
                                        "Status": status
                                    }])
        
                                elif cls == "2CL":
                                    rows = []
                                    for subj in ["AS", "NS", "AFS"]:
                                        grade = cadet.get(subj, "N/A")
                                        try:
                                            grade_val = float(grade)
                                            status = "Proficient" if grade_val >= 7 else "DEFICIENT"
                                        except:
                                            status = "N/A"
                                        rows.append({
                                            "Name": name_disp,
                                            "Subject": subj,
                                            "Grade": grade,
                                            "Status": status
                                        })
                                    df = pd.DataFrame(rows)
        
                                elif cls == "3CL":
                                    grade = cadet.get("MS231", "N/A")
                                    try:
                                        grade_val = float(grade)
                                        status = "Proficient" if grade_val >= 7 else "DEFICIENT"
                                    except:
                                        status = "N/A"
                                    df = pd.DataFrame([{
                                        "Name": name_disp,
                                        "Grade": grade,
                                        "Status": status
                                    }])
        
                                st.dataframe(df, hide_index=True)
            except Exception as e:
                st.error(f"Military tab error: {e}")
                
        with t5:
            try:
                conduct_sheet_map = {
                    "1CL": "1CL CONDUCT",
                    "2CL": "2CL CONDUCT",
                    "3CL": "3CL CONDUCT"
                }
                sheet_name = conduct_sheet_map.get(cls)
        
                if not sheet_name:
                    st.warning("Please select a valid class to view conduct data.")
                else:
                    conduct = sheet_df(sheet_name)
                    if conduct.empty:
                        st.info(f"No conduct data found for {sheet_name}.")
                    else:
                        conduct.columns = [c.strip().upper() for c in conduct.columns]
                        if "NAME" not in conduct.columns or "MERITS" not in conduct.columns:
                            st.error(f"Missing expected columns in {sheet_name}. Found: {conduct.columns.tolist()}")
                        else:
                            conduct["NAME_CLEANED"] = conduct["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                            cadet_data = conduct[conduct["NAME_CLEANED"] == name_clean]
        
                            if cadet_data.empty:
                                st.warning(f"No conduct data found for {name_disp} in {sheet_name}.")
                            else:
                                cadet_data = cadet_data.copy()
        
                                # First table: Merits and status
                                total_merits = cadet_data["MERITS"].astype(float).sum()
                                status = "Failed" if total_merits < 0 else "Passed"
                                merit_table = pd.DataFrame([{
                                    "Name": name_disp,
                                    "Merits": total_merits,
                                    "Status": status
                                }])
                                st.subheader("Merits Summary")
                                st.dataframe(merit_table, hide_index=True)
        
                                # Second table: Reports with Date and Class
                                report_table = cadet_data[["REPORTS", "DATE OF REPORT", "CLASS"]].copy()
                                report_table.columns = ["Reports", "Date of Report", "Class"]
                                st.subheader("Conduct Reports")
                                st.dataframe(report_table, hide_index=True)
        
            except Exception as e:
                st.error(f"Conduct tab error: {e}")

