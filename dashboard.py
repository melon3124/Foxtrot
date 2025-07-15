import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import re # Import regex for more robust cleaning

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
# Define scope and credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

try:
   creds_dict = st.secrets["google_service_account"]
    creds = Credentials.from_service_account_info(dict(creds_dict))
    client = gspread.authorize(creds)
    client = gspread.authorize(creds)
    SS = client.open("FOXTROT DASHBOARD V2")
except Exception as e:
    st.error(f"Error connecting to Google Sheets. Please check 'service_account.json' and sheet permissions: {e}")
    st.stop() # Stop the app if sheets connection fails

# -------------------- HELPERS --------------------
@st.cache_data(ttl=300) # Cache data for 5 minutes to reduce API calls
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans DataFrame columns by stripping whitespace and uppercasing."""
    df.columns = [c.strip().upper() for c in df.columns]
    return df

@st.cache_data(ttl=300) # Cache data for 5 minutes
def sheet_df(name: str) -> pd.DataFrame:
    """Fetches a worksheet and returns it as a cleaned Pandas DataFrame."""
    try:
        worksheet = SS.worksheet(name)
        return clean_df(pd.DataFrame(worksheet.get_all_records()))
    except gspread.exceptions.WorksheetNotFound:
        st.warning(f"Worksheet '{name}' not found in Google Sheet. Please check the sheet name.")
        return pd.DataFrame() # Return empty DataFrame on not found
    except Exception as e:
        st.error(f"Error fetching sheet '{name}': {e}")
        return pd.DataFrame() # Return empty DataFrame on other errors

def clean_cadet_name_for_comparison(name: str) -> str:
    """Standardizes a cadet name for robust comparison."""
    if not isinstance(name, str):
        return "" # Handle non-string inputs
    # Replace multiple spaces with a single space, then strip leading/trailing, then uppercase
    return re.sub(r'\s+', ' ', name).strip().upper()

# -------------------- LOAD DEMOGRAPHICS --------------------
demo_df = sheet_df("DEMOGRAPHICS")

if demo_df.empty:
    st.error("Demographics data could not be loaded. Please ensure your 'DEMOGRAPHICS' sheet exists and contains data.")
    st.stop() # Stop the app if demographics data is crucial and missing

# Create 'FULL NAME' for internal comparison and 'FULL NAME_DISPLAY' for UI
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
    demo_df["CLASS"] = "" # Initialize CLASS column if it doesn't exist

# Assign classes based on row index (1-based from problem description)
for cls, (start_idx, end_idx) in classes.items():
    # Adjusted for 0-based indexing for iloc and ensuring the range is inclusive
    # (start_idx - 1) because iloc is 0-indexed, end_idx because slice end is exclusive
    if not demo_df.empty:
        # Check if the column exists before trying to get its index, defensive coding
        if "CLASS" in demo_df.columns:
            demo_df.iloc[start_idx - 1:end_idx, demo_df.columns.get_loc("CLASS")] = cls
        else:
            st.warning("Warning: 'CLASS' column not found in demographics DataFrame for assignment.")
            break # Exit the loop if 'CLASS' column isn't there

# -------------------- SESSION STATE --------------------
if "mode" not in st.session_state:
    st.session_state.mode = "class"
if "selected_class" not in st.session_state:
    st.session_state.selected_class = None
if "selected_cadet_display_name" not in st.session_state:
    st.session_state.selected_cadet_display_name = None # For display
if "selected_cadet_cleaned_name" not in st.session_state:
    st.session_state.selected_cadet_cleaned_name = None # For comparison

# -------------------- TOP CONTROL BAR --------------------
st.markdown('<div class="centered">', unsafe_allow_html=True)
# Ensure the selectbox reflects the current session state if already selected
initial_select_index = 0
if st.session_state.selected_class:
    try:
        initial_select_index = ["", *classes.keys()].index(st.session_state.selected_class)
    except ValueError:
        pass # Keep 0 if not found

selected_class_from_select = st.selectbox("Select Class Level", ["", *classes.keys()], index=initial_select_index)

# Only update session state if a new selection is made or it's initial load
if selected_class_from_select != st.session_state.selected_class:
    st.session_state.mode = "class"
    st.session_state.selected_class = selected_class_from_select
    # Reset selected cadet when class changes to avoid displaying old data
    st.session_state.selected_cadet_display_name = None
    st.session_state.selected_cadet_cleaned_name = None
    # Rerun the app to apply the new class selection immediately
    st.rerun() 
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- CLASS (CADET) VIEW --------------------
if st.session_state.mode == "class" and st.session_state.selected_class:
    cls = st.session_state.selected_class
    cadets = demo_df[demo_df["CLASS"] == cls]

    if not cadets.empty:
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        cols_per_row = 4
        num_rows = (len(cadets) + cols_per_row - 1) // cols_per_row

        # Display buttons in a grid
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(cadets):
                    cadet_display_name = cadets.iloc[idx]["FULL NAME_DISPLAY"]
                    cadet_cleaned_name = cadets.iloc[idx]["FULL NAME"]
                    with cols[j]:
                        # Use a unique key for each button to avoid Streamlit warnings
                        if st.button(cadet_display_name, key=f"cadet_button_{cadet_cleaned_name}_{cls}"):
                            st.session_state.selected_cadet_display_name = cadet_display_name
                            st.session_state.selected_cadet_cleaned_name = cadet_cleaned_name
                            # Rerun to update the cadet details immediately
                            st.rerun() 
        st.markdown('</div>', unsafe_allow_html=True)

        # Retrieve selected cadet names from session state for display and comparison
        current_selected_cadet_display_name = st.session_state.selected_cadet_display_name
        current_selected_cadet_cleaned_name = st.session_state.selected_cadet_cleaned_name

        if current_selected_cadet_cleaned_name:
            # Get the row from demo_df using the cleaned name for lookup
            # This lookup is more robust as 'FULL NAME' is the cleaned version
            row = demo_df[demo_df["FULL NAME"] == current_selected_cadet_cleaned_name].iloc[0]

            st.markdown(f"## Showing details for: {current_selected_cadet_display_name}")

            t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "🏃 PFT", "📚 Academics", "🪖 Military", "⚖ Conduct"])

            with t1:
                pic, info = st.columns([1, 2])
                with pic:
                    # Use the display name for the image file path
                    img = f"profile_pics/{current_selected_cadet_display_name}.jpg"
                    st.image(img if os.path.exists(img) else "https://via.placeholder.com/400", width=350)
                with info:
                    left, right = st.columns(2)
                    # Filter out the internal 'FULL NAME' and 'FULL NAME_DISPLAY' columns
                    details = {k: v for k, v in row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]}
                    for idx, (k, v) in enumerate(details.items()):
                        (left if idx % 2 == 0 else right).write(f"**{k}:** {v}")

            with t2:
                try:
                    pft = sheet_df(f"{cls} PFT")
                    scale = sheet_df("SCALE_PFT")

                    if pft.empty or scale.empty:
                        st.info("PFT or SCALE_PFT data not available.")
                    else:
                        pft['NAME_CLEANED'] = pft["NAME"].astype(str).apply(clean_cadet_name_for_comparison)
                        r = pft[pft["NAME_CLEANED"] == current_selected_cadet_cleaned_name]

                        if not r.empty:
                            r = r.iloc[0]
                            gender = r.get("GENDER", "").upper()
                            class_level = cls

                            exercise_map = {
                                "PUSH-UPS": "Push-ups",
                                "SITUPS": "Sit-ups",
                                "PULL-UPS/ FLEX": "Pull-ups" if gender == "M" else "Flex Arm Hang",
                                "RUN": "3.2 KM Run"
                            }

                            column_map = {
                                "MALE": {
                                    "1CL": {
                                        "PUSH-UPS": ("PUSH-UPS_MALE_1CL", "GRADE_MALE_PUSHUPS_1CL"),
                                        "SITUPS": ("SITUPS_MALE_1CL", "GRADE_MALE_SITUPS_1CL"),
                                        "PULL-UPS/ FLEX": ("PULLUPS_MALE_1CL", "GRADE_MALE_PULLUPS_1CL"),
                                        "RUN": ("RUN_MALE_1CL", "GRADE_MALE_RUN_1CL")
                                    },
                                    "2CL": {
                                        "PUSH-UPS": ("PUSH-UPS_MALE_2CL", "GRADE_MALE_PUSHUPS_2CL"),
                                        "SITUPS": ("SITUPS_MALE_2CL", "GRADE_MALE_SITUPS_2CL"),
                                        "PULL-UPS/ FLEX": ("PULLUPS_MALE_2CL", "GRADE_MALE_PULLUPS_2CL"),
                                        "RUN": ("RUN_MALE_2CL", "GRADE_MALE_RUN_2CL")
                                    },
                                    "3CL": {
                                        "PUSH-UPS": ("PUSH-UPS_MALE_3CL", "GRADE_MALE_PUSHUPS_3CL"),
                                        "SITUPS": ("SITUPS_MALE_3CL", "GRADE_MALE_SITUPS_3CL"),
                                        "PULL-UPS/ FLEX": ("PULLUPS_MALE_3CL", "GRADE_MALE_PULLUPS_3CL"),
                                        "RUN": ("RUN_MALE_3CL", "GRADE_MALE_RUN_3CL")
                                    }
                                },
                                "FEMALE": {
                                    "1CL": {
                                        "PUSH-UPS": ("PUSH-UPS_FEMALE_1CL", "GRADE_FEMALE_PUSHUPS_1CL"),
                                        "SITUPS": ("SITUPS_FEMALE_1CL", "GRADE_FEMALE_SITUPS_1CL"),
                                        "PULL-UPS/ FLEX": ("FLEX_FEMALE_1CL", "GRADE_FEMALE_FLEX_1CL"),
                                        "RUN": ("RUN_FEMALE_1CL", "GRADE_FEMALE_RUN_1CL")
                                    },
                                    "2CL": {
                                        "PUSH-UPS": ("PUSH-UPS_FEMALE_2CL", "GRADE_FEMALE_PUSHUPS_2CL"),
                                        "SITUPS": ("SITUPS_FEMALE_2CL", "GRADE_FEMALE_SITUPS_2CL"),
                                        "PULL-UPS/ FLEX": ("FLEX_FEMALE_2CL", "GRADE_FEMALE_FLEX_2CL"),
                                        "RUN": ("RUN_FEMALE_2CL", "GRADE_FEMALE_RUN_2CL")
                                    },
                                    "3CL": {
                                        "PUSH-UPS": ("PUSH-UPS_FEMALE_3CL", "GRADE_FEMALE_PUSHUPS_3CL"),
                                        "SITUPS": ("SITUPS_FEMALE_3CL", "GRADE_FEMALE_SITUPS_3CL"),
                                        "PULL-UPS/ FLEX": ("FLEX_FEMALE_3CL", "GRADE_FEMALE_FLEX_3CL"),
                                        "RUN": ("RUN_FEMALE_3CL", "GRADE_FEMALE_RUN_3CL")
                                    }
                                }
                            }

                            results = []
                            for col, label in exercise_map.items():
                                raw_score = r.get(col, "")
                                try:
                                    score_val = float(raw_score)
                                except:
                                    results.append((label, raw_score, "N/A", "N/A"))
                                    continue

                                try:
                                    range_col, grade_col = column_map["MALE" if gender == "M" else "FEMALE"][class_level][col]
                                except KeyError:
                                    results.append((label, score_val, "N/A", "Mapping Error"))
                                    continue

                                matched_grade = "N/A"
                                for _, row_s in scale.iterrows():
                                    try:
                                        range_expr = str(row_s[range_col]).strip()
                                        if "-" in range_expr:
                                            low, high = map(float, range_expr.split("-"))
                                            if low <= score_val <= high:
                                                matched_grade = row_s[grade_col]
                                                break
                                        elif ">=" in range_expr and score_val >= float(range_expr.replace(">=", "")):
                                            matched_grade = row_s[grade_col]
                                            break
                                        elif "<=" in range_expr and score_val <= float(range_expr.replace("<=", "")):
                                            matched_grade = row_s[grade_col]
                                            break
                                        elif ">" in range_expr and score_val > float(range_expr.replace(">", "")):
                                            matched_grade = row_s[grade_col]
                                            break
                                        elif "<" in range_expr and score_val < float(range_expr.replace("<", "")):
                                            matched_grade = row_s[grade_col]
                                            break
                                    except:
                                        continue

                                try:
                                    numeric_grade = float(matched_grade)
                                    status = "Full Duty" if numeric_grade >= 8 else "Not Full Duty"
                                except:
                                    status = "N/A"

                                results.append((label, score_val, matched_grade, status))

                            df = pd.DataFrame(results, columns=["Exercise", "Raw Score", "Equivalent Grade", "Status"])
                            st.dataframe(df, hide_index=True)
                        else:
                            st.warning(f"No PFT record found for {current_selected_cadet_display_name}.")
                except Exception as e:
                    st.error(f"PFT load error: {e}")


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
