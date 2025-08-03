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
    
# Add these lines to initialize session state variables
if "mode" not in st.session_state:
    st.session_state["mode"] = "class"
if "selected_class" not in st.session_state:
    st.session_state["selected_class"] = ""
if "selected_cadet_display_name" not in st.session_state:
    st.session_state["selected_cadet_display_name"] = None
if "selected_cadet_cleaned_name" not in st.session_state:
    st.session_state["selected_cadet_cleaned_name"] = None

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
    st.set_page_config(layout="wide")
    st.title("📊 Summary Dashboard")

    term = st.selectbox("Select Term", ["1st Term", "2nd Term"], key="summary_term")
    selected_class = st.selectbox("Select Class", ["1CL", "2CL", "3CL"], key="summary_class")

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

    acad_tab, pft_tab, mil_tab, conduct_tab = st.tabs(["📚 Academics", "🏃 PFT", "🪦 Military", "⚖ Conduct"])

    with acad_tab:
        st.subheader("📚 Academic Summary")
        sheet_name = acad_hist_map[selected_class][term]
        acad_df = sheet_df(sheet_name)
    
        if not acad_df.empty:
            acad_df.columns = [c.strip().upper() for c in acad_df.columns]  # Normalize headers
    
            if "NAME" not in acad_df.columns:
                st.warning(f"❗ 'NAME' column missing in {sheet_name}. Check sheet headers.")
            else:
                st.markdown(f"## 🎓 {selected_class} Academic Performance")
                acad_df["NAME_CLEANED"] = acad_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                subject_cols = [col for col in acad_df.columns if col not in ["NAME", "NAME_CLEANED"]]
    
                for subject in subject_cols:
                    acad_df[subject] = pd.to_numeric(acad_df[subject], errors='coerce')
                    prof = acad_df[acad_df[subject] >= 7][["NAME", subject]].dropna().sort_values(by=subject, ascending=False)
                    defn = acad_df[acad_df[subject] < 7][["NAME", subject]].dropna().sort_values(by=subject)
                    max_def = defn.sort_values(by=subject).head(1) if not defn.empty else pd.DataFrame()
    
                    with st.expander(f"📘 Subject: **{subject}** ({len(prof)} ✅ / {len(defn)} 🚫)"):
                        col1, col2 = st.columns(2)
    
                        with col1:
                            st.markdown("**✅ Proficient Cadets**")
                            if not prof.empty:
                                st.dataframe(prof, use_container_width=True, hide_index=True)
                            else:
                                st.info("No proficient cadets.")
    
                        with col2:
                            st.markdown("**🚫 Deficient Cadets**")
                            if not defn.empty:
                                st.dataframe(defn, use_container_width=True, hide_index=True)
                            else:
                                st.success("No deficiencies recorded.")
    
                        if not max_def.empty:
                            st.markdown("⬇️ **Cadet with Highest Deficiency**")
                            st.dataframe(max_def, use_container_width=True, hide_index=True)
        else:
            st.warning(f"⚠️ No data found in {sheet_name}.")


    with pft_tab:
        st.subheader("🏃 PFT Summary")
        sheet_name = pft_sheet_map[selected_class][term]
        pft_df = sheet_df(sheet_name)
        if not pft_df.empty:
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
            if not top_male.empty:
                st.dataframe(top_male[["NAME", "AVG_GRADE"]], use_container_width=True)
            else:
                st.warning("No male cadet data available.")

            st.write("💪 Strongest Female Cadet")
            if not top_female.empty:
                st.dataframe(top_female[["NAME", "AVG_GRADE"]], use_container_width=True)
            else:
                st.warning("No female cadet data available.")
                            
    with mil_tab:
        st.subheader("🫦 Military Summary")
        sheet_name = mil_sheet_map[selected_class][term]
        mil_df = sheet_df(sheet_name)
        if not mil_df.empty:
            st.markdown(f"### {selected_class} Military Grades")
            if selected_class == "1CL":
                mil_df["GRADE"] = pd.to_numeric(mil_df["GRADE"], errors="coerce")
                prof = mil_df[mil_df["GRADE"] >= 7]
                defn = mil_df[mil_df["GRADE"] < 7]
            elif selected_class == "2CL":
                for col in ["AS", "NS", "AFS"]:
                    mil_df[col] = pd.to_numeric(mil_df[col], errors="coerce")
                prof = mil_df[(mil_df["AS"] >= 7) & (mil_df["NS"] >= 7) & (mil_df["AFS"] >= 7)]
                defn = mil_df[(mil_df["AS"] < 7) | (mil_df["NS"] < 7) | (mil_df["AFS"] < 7)]
            elif selected_class == "3CL":
                mil_df["MS231"] = pd.to_numeric(mil_df["MS231"], errors="coerce")
                prof = mil_df[mil_df["MS231"] >= 7]
                defn = mil_df[mil_df["MS231"] < 7]

            st.write("✅ Proficient Cadets")
            st.dataframe(prof[["NAME"]], use_container_width=True)
            st.write("🚫 Deficient Cadets")
            st.dataframe(defn[["NAME"]], use_container_width=True)

    with conduct_tab:
        st.subheader("⚖ Conduct Summary")
        sheet_name = conduct_sheet_map[selected_class][term]
        conduct_df = sheet_df(sheet_name)
        if not conduct_df.empty:
            st.markdown(f"### {selected_class} Conduct Summary")
            conduct_df.columns = [c.strip().upper() for c in conduct_df.columns]
            conduct_df["MERITS"] = pd.to_numeric(conduct_df.get("MERITS", 0), errors='coerce')

            touring_yes_df = conduct_df[conduct_df.get("TOURING STATUS", "NO").str.upper() == "YES"]
            flagged_df = conduct_df[conduct_df["MERITS"] < 20]

            st.write("🎒 Touring Cadets")
            st.dataframe(touring_yes_df[["NAME", "TOURING STATUS"]], use_container_width=True)

            st.write("🔴 Cadets with < 20 Merits")
            st.dataframe(flagged_df[["NAME", "MERITS"]], use_container_width=True)

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
                    k: v for k, v in row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]
                }
                # Iterate through the items and place them in alternating columns
                for idx, (k, v) in enumerate(info_to_display.items()):
                    if idx % 2 == 0:
                        col1.markdown(f"**{k}**: {v}")
                    else:
                        col2.markdown(f"**{k}**: {v}")

    # --- ACADEMICS TAB (T2) ---
    with t2:
        try:
            st.header("Academics")
            st.caption(f"Showing academic data for **:red[{name_disp}]**")
            
            # Placeholder for get_data_from_gsheet and update_sheet_cell
            def get_data_from_gsheet(sheet_name):
                # Dummy data for demonstration
                if sheet_name == "ACADEMICS":
                    return pd.DataFrame({
                        "NAME": [name_disp, name_disp],
                        "SUBJECT": ["Math", "Science"],
                        "TERM": ["1st Term", "1st Term"],
                        "GRADE": [9.0, 8.5]
                    })
                if sheet_name == "ACADEMIC_HISTORY":
                    return pd.DataFrame({
                        "NAME": [name_disp, name_disp],
                        "SUBJECT": ["Math", "Science"],
                        "TERM": ["1st Term", "1st Term"],
                        "GRADE": [8.0, 8.0]
                    })
                return pd.DataFrame()

            def update_sheet_cell(sheet_name, name_to_find, col_to_find, val_to_find, col_to_update, new_value):
                st.success(f"Mock update to {sheet_name}: {name_to_find}'s {col_to_update} for {col_to_find}={val_to_find} changed to {new_value}")
                return True

            def evaluate_status(grade):
                try:
                    val = float(grade)
                    return "Proficient" if val >= 7 else "DEFICIENT"
                except:
                    return "N/A"

            df_curr_grades = get_data_from_gsheet("ACADEMICS")
            df_history_grades = get_data_from_gsheet("ACADEMIC_HISTORY")

            selected_term = st.selectbox(
                "Select Term",
                options=df_curr_grades["TERM"].unique().tolist() if not df_curr_grades.empty else ["1st Term"],
                key="acad_term_selector"
            )
            
            with st.expander("Academic Record", expanded=True):
                if not df_curr_grades.empty and not df_history_grades.empty:
                    df_curr_grades_filtered = df_curr_grades[
                        (df_curr_grades["NAME"] == name_disp) & 
                        (df_curr_grades["TERM"] == selected_term)
                    ].copy()

                    df_history_grades_filtered = df_history_grades[
                        (df_history_grades["NAME"] == name_disp) & 
                        (df_history_grades["TERM"] == selected_term)
                    ].copy()

                    df_to_display = pd.merge(
                        df_history_grades_filtered,
                        df_curr_grades_filtered,
                        on=["NAME", "SUBJECT", "TERM"],
                        how="outer",
                        suffixes=("_hist", "")
                    )
                    
                    df_to_display = df_to_display.rename(columns={
                        "GRADE_hist": "PREVIOUS GRADE",
                        "GRADE": "CURRENT GRADE"
                    })

                    df_to_display["CURRENT GRADE"] = pd.to_numeric(df_to_display["CURRENT GRADE"], errors='coerce')
                    df_to_display["PREVIOUS GRADE"] = pd.to_numeric(df_to_display["PREVIOUS GRADE"], errors='coerce')

                    df_to_display["Increase/Decrease"] = (
                        df_to_display["CURRENT GRADE"] - df_to_display["PREVIOUS GRADE"]
                    ).fillna(0)
                    
                    df_to_display["Status"] = df_to_display["CURRENT GRADE"].apply(evaluate_status)

                    df_to_display = df_to_display[[
                        "NAME", "SUBJECT", "PREVIOUS GRADE", "CURRENT GRADE", "Increase/Decrease", "Status"
                    ]].copy()

                    with st.form("edit_acad_form"):
                        st.markdown("### Edit Academic Grades")
                        
                        gb = GridOptionsBuilder.from_dataframe(df_to_display)
                        gb.configure_column("SUBJECT", editable=False)
                        gb.configure_column("PREVIOUS GRADE", editable=False)
                        gb.configure_column("CURRENT GRADE", editable=True)
                        gb.configure_column("Increase/Decrease", editable=False)
                        gb.configure_column("Status", editable=False)
                        
                        go = gb.build()
                        
                        grid_response = AgGrid(
                            df_to_display,
                            gridOptions=go,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            allow_unsafe_jscode=True,
                            theme="streamlit",
                            height=300,
                            key="acad_grid"
                        )
                        
                        edited_df = grid_response["data"]

                        submitted = st.form_submit_button("📤 Submit Changes")
                
                    if submitted:
                        try:
                            for index, row in edited_df.iterrows():
                                update_sheet_cell(
                                    sheet_name="ACADEMICS",
                                    name_to_find=name_disp,
                                    col_to_find="SUBJECT",
                                    val_to_find=row["SUBJECT"],
                                    col_to_update="GRADE",
                                    new_value=str(row["CURRENT GRADE"])
                                )
                                update_sheet_cell(
                                    sheet_name="ACADEMIC_HISTORY",
                                    name_to_find=name_disp,
                                    col_to_find="SUBJECT",
                                    val_to_find=row["SUBJECT"],
                                    col_to_update="GRADE",
                                    new_value=str(row["CURRENT GRADE"])
                                )
                            st.cache_data.clear()
                            time.sleep(1)
                            st.success("✅ Academic grades updated successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error submitting academic data: {e}")
                else:
                    st.warning("No academic data found for this cadet.")
        except Exception as e:
            st.error(f"❌ Unexpected error in Academics tab: {e}")


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
                    # Merits Editor
                    st.subheader("📜 Merits Summary")
                    current_merits = cadet_data.iloc[0].get("merits", "0")
                    merits_value = st.number_input(
                        f"Edit Merits – {term}",
                        value=float(current_merits) if str(current_merits).replace('.', '', 1).lstrip('-').isdigit() else 0.0,
                        step=1.0
                    )
                    status = "🚫 Failed" if merits_value < 0 else "✅ Passed"
                    st.dataframe(pd.DataFrame([{
                        "Name": name_disp, "Merits": merits_value, "Status": status
                    }]), hide_index=True, use_container_width=True)
    
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
    
                    # Conduct Reports
                    st.subheader("📋 Conduct Reports")
                    expected_cols = ["NAME", "REPORT", "DATE OF REPORT", "NATURE", "DEMERITS"]
                    reports_df = sheet_df("REPORTS")
                    reports_df.columns = [c.strip().upper() for c in reports_df.columns]
                    reports_df["NAME_CLEANED"] = reports_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                    cadet_reports = reports_df[reports_df["NAME_CLEANED"] == name_clean].copy()
    
                    st.dataframe(
                        cadet_reports[["NAME", "REPORT", "DATE OF REPORT", "NATURE", "DEMERITS"]],
                        use_container_width=True,
                        hide_index=True
                    )
    
                    # Touring Status Editor
                    st.subheader("🧭 Touring Status")
                    touring_status = cadet_data.iloc[0].get("touring status", "").strip().capitalize()
                    current_touring = "Yes" if touring_status.lower() == "yes" else "No"
                    new_touring = st.selectbox(
                        "TOURING?",
                        options=["Yes", "No"],
                        index=0 if current_touring == "Yes" else 1,
                        key=f"touring_status_selectbox_{name_clean}"
                    )
    
                    if st.button(f"💾 Save Touring Status – {term}", key=f"save_touring_status_{name_clean}"):
                        try:
                            full_df = sheet_df(sheet_name)
                            full_df.columns = [c.strip().lower() for c in full_df.columns]
                            full_df["name_cleaned"] = full_df["name"].astype(str).apply(clean_cadet_name_for_comparison)
                            full_df.loc[full_df["name_cleaned"] == name_clean, "touring status"] = new_touring
                            full_df.drop(columns=["name_cleaned"], inplace=True)
                            update_sheet(sheet_name, full_df)
                            sheet_df.clear()
                            st.success("✅ Touring status updated successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to update touring status: {e}")
    
                    # Add New Report
                    st.subheader("➕ Add New Conduct Report")
                    with st.form("report_form"):
                        new_report = st.text_area("Report Description", placeholder="Enter behavior details...")
                        new_report_date = st.date_input("Date of Report")
                        new_nature = st.selectbox("Nature", ["I", "II", "III", "IV"])
                        new_demerits = st.number_input("Demerits", step=1)
                        submitted = st.form_submit_button("📤 Submit Report")
    
                    if submitted:
                        try:
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

