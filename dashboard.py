import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import re
import unicodedata

# -------------------- CONFIG --------------------
st.set_page_config(layout="wide")

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

# -------------------- HELPERS --------------------
def clean_grade(grade_raw: str) -> str:
    if not isinstance(grade_raw, str):
        grade_raw = str(grade_raw)
    normalized = unicodedata.normalize("NFKD", grade_raw)
    cleaned = (
        normalized.replace("\xa0", "")
                  .replace("\u202f", "")
                  .replace("\u2009", "")
                  .replace("\u200a", "")
                  .replace(" ", "")
                  .replace("%", "")
                  .replace(" ", "")
                  .strip()
    )
    return cleaned

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
        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "🏃 PFT", "📚 Academics", "🪖 Military", "⚖ Conduct"])

        with t1:
            pic, info = st.columns([1, 2])
            with pic:
                img_path = f"profile_pics/{name_disp}.jpg"
                st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
            with info:
                left, right = st.columns(2)
                for idx, (k, v) in enumerate({k: v for k, v in row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]}.items()):
                    (left if idx % 2 == 0 else right).write(f"**{k}:** {v}")

        with t2:
            pft_df = sheet_df(f"{cls} PFT")
            if not pft_df.empty:
                pft_df["NAME_CLEANED"] = pft_df["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                cadet_row = pft_df[pft_df["NAME_CLEANED"] == name_clean]
                if not cadet_row.empty:
                    cadet = cadet_row.iloc[0]
                    exercises = [
                        ("Push-ups", "PUSH-UPS", "PUSHUPS_GRADE"),
                        ("Sit-ups", "SITUPS", "SITUPS_GRADE"),
                        ("Pull-ups / Flex", "PULL-UPS/ FLEX", "PULL-UPS/ FLEX_GRADE"),
                        ("3.2km Run", "RUN", "RAN_GRADE")
                    ]
                    results = []
                    for label, raw_col, grade_col in exercises:
                        raw = cadet.get(raw_col, "")
                        grade = clean_grade(cadet.get(grade_col, ""))
                        try:
                            status = "Proficient" if float(grade) >= 7 else "Deficient"
                        except:
                            status = "N/A"
                        results.append({"Exercise": label, "Repetitions / Time": raw, "Grade": grade, "Status": status})
                    st.markdown("### PFT Breakdown")
                    st.dataframe(pd.DataFrame(results), hide_index=True)

        # Tabs t3 (Academics), t4 (Military), t5 (Conduct) unchanged from original; you may paste similarly with normalize_column_name applied
        
            with t3: # Academics tab - main focus of the fix
                try:
                    acad_sheet_map = {
                        "1CL": "1CL ACAD",
                        "2CL": "2CL ACAD",
                        "3CL": "3CL ACAD"
                    }
                    acad = sheet_df(acad_sheet_map[cls]) # Already cleaned by sheet_df

                    if acad.empty:
                        st.info("No Academic data available for this class.")
                    else:
                        # Consistently look for "NAME" column in academic sheets
                        target_name_col = "NAME" 

                        if target_name_col not in acad.columns:
                            st.error(f"Error: Expected column '{target_name_col}' not found in the academic sheet '{acad_sheet_map[cls]}'.")
                            st.write(f"Available columns in '{acad_sheet_map[cls]}': {acad.columns.tolist()}")
                            # No return here, just display error and skip remaining logic for this tab
                        else: # Only proceed if the target_name_col exists
                            # Apply robust cleaning to the academic sheet's name column
                            acad['NAME_CLEANED'] = acad[target_name_col].astype(str).apply(clean_cadet_name_for_comparison)

                            r = acad[acad["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                            if not r.empty:
                                r = r.iloc[0]
                                # Drop the original name column and the cleaned name column before displaying
                                # Ensure 'NAME' is dropped if it's the target column for consistency
                                df_data = r.drop([col for col in r.index if col in [target_name_col, 'NAME_CLEANED']], errors='ignore')
                                
                                df = pd.DataFrame({"Subject": df_data.index, "Grade": df_data.values})
                                
                                df["Grade_Numeric"] = pd.to_numeric(df["Grade"], errors='coerce')
                                df["Status"] = df["Grade_Numeric"].apply(lambda g: "Proficient" if g >= 7 else "Deficient" if pd.notna(g) else "N/A")
                                
                                st.dataframe(df[['Subject', 'Grade', 'Status']], hide_index=True)
                            else:
                                st.warning(f"No academic record found for {current_selected_cadet_display_name}.")
                                # Optional: For debugging, uncomment to see available cleaned names
                                # st.write("Available names in sheet (cleaned):", acad['NAME_CLEANED'].tolist())
                except Exception as e:
                    st.error(f"Academic load error: {e}")
                    # import traceback
                    # st.error(traceback.format_exc()) # Uncomment for full traceback

            with t4:
                try:
                    mil = sheet_df(f"{cls} MIL")
                    if mil.empty:
                        st.info("No Military data available for this class.")
                    else:
                        mil['NAME_CLEANED'] = mil["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                        r = mil[mil["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                        if not r.empty:
                            # Drop 'NAME' and 'NAME_CLEANED' from items before iterating
                            for subj, grade in r.iloc[0].drop(['NAME', 'NAME_CLEANED'], errors='ignore').items():
                                st.write(f"**{subj}:** {grade}")
                        else:
                            st.info("No Military data available for this cadet.")
                except Exception as e:
                    st.error(f"Military load error: {e}")
                    # import traceback
                    # st.error(traceback.format_exc())

            with t5:
                try:
                    cond = sheet_df("CONDUCT")
                    if cond.empty:
                        st.info("No Conduct data available.")
                    else:
                        cond['NAME_CLEANED'] = cond["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                        r = cond[cond["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                        if not r.empty:
                            # Drop 'NAME' and 'NAME_CLEANED' from items before iterating
                            for subj, val in r.iloc[0].drop(['NAME', 'NAME_CLEANED'], errors='ignore').items():
                                st.write(f"**{subj}:** {val}")
                        else:
                            st.info("No Conduct data available for this cadet.")
                except Exception as e:
                    st.error(f"Conduct load error: {e}")
                    # import traceback
                    # st.error(traceback.format_exc())
    else:
        st.warning(f"No cadets found for class {cls}.")
