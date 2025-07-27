import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import os
import re
import unicodedata
import time
import json
# Removed: import pygsheets # Keeping this import, though primarily using gspread for updates
# Removed: from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode

# --- Session State Initialization ---
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "t3" # Default to PFT tab after refresh if it was the last active

# --- Helper Functions ---
def clean_column_names(df):
    """Cleans DataFrame column names by stripping, uppercasing."""
    # Ensure all column names are strings before applying .strip() and .upper()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def evaluate_status(grade):
    """Evaluates proficiency status based on a numerical grade."""
    try:
        val = float(grade)
        return "PROFICIENT" if val >= 7 else "DEFICIENT" # Changed to match string used in CSS
    except (ValueError, TypeError):
        return "N/A"

def clean_cadet_name_for_comparison(name):
    """Normalizes cadet names for consistent comparison."""
    if not isinstance(name, str):
        return ""
    # Remove extra spaces, strip, and convert to uppercase
    name = re.sub(r'\s+', ' ', name).strip().upper()
    # Normalize Unicode characters (e.g., é -> e)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    return name

def normalize_column_name(col: str) -> str:
    """Normalizes column names for consistency."""
    if not isinstance(col, str):
        col = str(col)
    col = unicodedata.normalize("NFKD", col)
    # Remove specific non-breaking spaces and other peculiar unicode spaces
    col = col.replace("\xa0", "").replace("\u202f", "").replace("\u2009", "").replace(" ", "")
    col = re.sub(r'[^\w\-/ ]+', '', col).strip() # Allow hyphens and slashes
    return col.upper()

# --- CSS Class Helper for st.data_editor Status Column ---
def get_status_css_class(row_data):
    """
    Returns a CSS class name based on the 'Status' value in the row.
    This function is called by st.data_editor for each cell in the 'Status' column.
    """
    status_value = row_data["Status"] # Access the 'Status' key from the row data
    if status_value == 'DEFICIENT':
        return "status-deficient"
    elif status_value == 'PROFICIENT':
        return "status-proficient"
    return "" # No special class for N/A or others

# --- Google Sheets Authorization (moved up for early access) ---
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

try:
    credentials = Credentials.from_service_account_info(
        st.secrets["google_service_account"],
        scopes=scopes
    )
    gc = gspread.authorize(credentials)
    SS = gc.open("FOXTROT DASHBOARD V2")
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}")
    st.stop()

# --- Centralized Sheet Update Function ---
def update_google_sheet(sheet_name, dataframe_to_write):
    """
    Updates a specified Google Sheet worksheet with the provided DataFrame.
    Clears the sheet and writes the new data, then clears Streamlit's cache.
    """
    try:
        worksheet = SS.worksheet(sheet_name)
        with st.spinner(f"Updating '{sheet_name}' sheet..."): # Use st.spinner as a context manager
            worksheet.clear()
            # Convert DataFrame to list of lists, including headers
            data_to_send = [dataframe_to_write.columns.values.tolist()] + dataframe_to_write.values.tolist()
            worksheet.update("A1", data_to_send)
        st.toast(f"✅ Google Sheet '{sheet_name}' updated successfully!", icon="🎉")
        st.cache_data.clear() # Clear cache after any write operation
        time.sleep(0.5) # Give toast time to show
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"❌ Worksheet '{sheet_name}' not found. Please check the sheet name.")
    except Exception as e:
        st.error(f"❌ Failed to update Google Sheet '{sheet_name}': {e}")


@st.cache_data(ttl=300) # Cache sheet data for 5 minutes
def sheet_df(name: str) -> pd.DataFrame:
    """Fetches and cleans data from a specified Google Sheet worksheet."""
    try:
        worksheet = SS.worksheet(name)
        df = pd.DataFrame(worksheet.get_all_records())
        return clean_column_names(df)
    except gspread.exceptions.WorksheetNotFound:
        st.warning(f"Worksheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching sheet '{name}': {e}")
        return pd.DataFrame()

# -------------------- Login Logic --------------------
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
    st.cache_data.clear() # Clear cache on logout
    st.rerun()

# -------------------- CONFIG --------------------
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
        /* Custom CSS for data_editor status cells */
        .status-deficient { /* Using a custom class name for deficient status */
            background-color: #FFCCCC; /* Light red */
            color: black !important; /* Ensure text is readable */
        }
        .status-proficient { /* Using a custom class name for proficient status */
            background-color: #CCFFCC; /* Light green */
            color: black !important; /* Ensure text is readable */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1>🦊 Welcome to Foxtrot Company CIS</h1>", unsafe_allow_html=True)

# -------------------- DEMOGRAPHICS --------------------
demo_df = sheet_df("DEMOGRAPHICS")
if demo_df.empty:
    st.error("Demographics sheet missing or empty.")
    st.stop()

# Ensure 'FAMILY NAME', 'FIRST NAME', 'MIDDLE NAME', 'EXTN' columns exist for safety
for col_name in ['FAMILY NAME', 'FIRST NAME', 'MIDDLE NAME', 'EXTN']:
    if col_name.upper() not in [c.upper() for c in demo_df.columns]:
        demo_df[col_name.upper()] = "" # Add in uppercase to match clean_column_names

# Ensure the correct case for column access after cleaning
demo_df.columns = [c.upper() for c in demo_df.columns]

demo_df["FULL NAME"] = demo_df.apply(
    lambda r: clean_cadet_name_for_comparison(
        f"{r.get('FAMILY NAME', '')}, {r.get('FIRST NAME', '')} {r.get('MIDDLE NAME', '')} {r.get('EXTN', '')}"
    ), axis=1
)
demo_df["FULL NAME_DISPLAY"] = demo_df.apply(
    lambda r: f"{r.get('FAMILY NAME', '')}, {r.get('FIRST NAME', '')} {r.get('MIDDLE NAME', '')} {r.get('EXTN', '')}".strip(), axis=1
)

classes = {
    "1CL": (2, 27),
    "2CL": (30, 61),
    "3CL": (63, 104)
}
if "CLASS" not in demo_df.columns:
    demo_df["CLASS"] = ""
for cls, (start, end) in classes.items():
    # Adjusting for 0-based indexing for iloc
    demo_df.loc[demo_df.index[start-1:end], "CLASS"] = cls


# -------------------- SESSION STATE FOR NAVIGATION --------------------
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

# -------------------- CLASS VIEW / CADET DETAILS --------------------
cls = st.session_state.selected_class
if st.session_state.mode == "class" and cls:
    cadets = demo_df[demo_df["CLASS"] == cls]
    if cadets.empty:
        st.warning(f"No cadets found for class {cls}.")
    else:
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        # Display cadet buttons in columns
        num_cols = 4
        cols = st.columns(num_cols)
        for i, (idx, cadet_row) in enumerate(cadets.iterrows()):
            name_display = cadet_row["FULL NAME_DISPLAY"]
            name_cleaned = cadet_row["FULL NAME"]
            with cols[i % num_cols]:
                if st.button(name_display, key=f"cadet_{name_cleaned}_{cls}"):
                    st.session_state.selected_cadet_display_name = name_display
                    st.session_state.selected_cadet_cleaned_name = name_cleaned
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    name_disp = st.session_state.selected_cadet_display_name
    name_clean = st.session_state.selected_cadet_cleaned_name

    if name_clean:
        # Fetch the most up-to-date cadet row
        cadet_info_row_opt = demo_df[demo_df["FULL NAME"] == name_clean]
        if cadet_info_row_opt.empty:
            st.error(f"Could not find detailed information for cadet: {name_disp}")
            st.stop() # Stop execution if cadet info isn't found
        cadet_info_row = cadet_info_row_opt.iloc[0]


        t1, t2, t3, t4, t5 = st.tabs(["👤 Demographics", "📚 Academics", "🏃 PFT", "🪖 Military", "⚖ Conduct"])

        # --- Tab 1: Demographics ---
        with t1:
            pic, info = st.columns([1, 2])
            with pic:
                img_path = f"profile_pics/{name_disp}.jpg"
                st.image(img_path if os.path.exists(img_path) else "https://via.placeholder.com/400", width=350)
            with info:
                left, right = st.columns(2)
                # Filter out the internal 'FULL NAME' columns and 'CLASS'
                display_data = {k: v for k, v in cadet_info_row.items() if k not in ["FULL NAME", "FULL NAME_DISPLAY", "CLASS"]}
                for idx, (k, v) in enumerate(display_data.items()):
                    (left if idx % 2 == 0 else right).write(f"**{k}:** {v}")

        # --- Tab 2: Academics ---
        with t2:
            if "selected_acad_term" not in st.session_state:
                st.session_state.selected_acad_term = "1st Term"

            term = st.radio(
                "Select Term",
                ["1st Term", "2nd Term"],
                index=["1st Term", "2nd Term"].index(st.session_state.selected_acad_term),
                horizontal=True,
                help="Choose academic term for grade comparison",
                key="acad_term_radio"
            )
            st.session_state.selected_acad_term = term

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

            def find_name_column(df_columns):
                """Finds a name column among possible variations."""
                for col in possible_name_cols:
                    if col.upper() in [c.upper() for c in df_columns]:
                        # Return the exact column name as it appears in the df
                        return next((c for c in df_columns if c.upper() == col.upper()), None)
                return None

            try:
                prev_sheet_name = acad_sheet_map[cls][term]
                curr_sheet_name = acad_hist_map[cls][term]

                prev_df_raw = sheet_df(prev_sheet_name)
                curr_df_raw = sheet_df(curr_sheet_name)

                # If prev_df_raw is empty, create a dummy to prevent errors
                if prev_df_raw.empty:
                    st.warning(f"⚠️ No valid previous academic data found in '{prev_sheet_name}'.")
                    prev_df_raw = pd.DataFrame(columns=['NAME'])

                # If curr_df_raw is empty, create a dummy to prevent errors
                if curr_df_raw.empty:
                    st.info(f"⚠️ No valid current academic data found in '{curr_sheet_name}'. A new row will be created upon submission.")
                    curr_df_raw = pd.DataFrame(columns=['NAME'])

                # Normalize columns for consistency across both DataFrames
                prev_df = clean_column_names(prev_df_raw.copy())
                curr_df = clean_column_names(curr_df_raw.copy())

                prev_name_col = find_name_column(prev_df.columns)
                curr_name_col = find_name_column(curr_df.columns)

                # Ensure name columns exist in the processed DFs
                if prev_name_col is None:
                    prev_df.insert(0, "NAME", "")
                    prev_name_col = "NAME"
                if curr_name_col is None:
                    curr_df.insert(0, "NAME", "")
                    curr_name_col = "NAME"

                # Add temporary cleaned name columns for merging/lookup
                prev_df["_TEMP_NAME_CLEANED"] = prev_df[prev_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                curr_df["_TEMP_NAME_CLEANED"] = curr_df[curr_name_col].astype(str).apply(clean_cadet_name_for_comparison)

                # Get cadet's previous academic data
                row_prev_opt = prev_df[prev_df["_TEMP_NAME_CLEANED"] == name_clean]
                row_prev = row_prev_opt.iloc[0] if not row_prev_opt.empty else pd.Series(dtype='object')

                # Get cadet's current academic data
                row_curr_opt = curr_df[curr_df["_TEMP_NAME_CLEANED"] == name_clean]
                row_curr = row_curr_opt.iloc[0] if not row_curr_opt.empty else pd.Series(dtype='object')

                # Identify all unique subject columns from both previous and current data
                # Exclude internal columns and potential existing 'STATUS' or 'PROFICIENCY/DEFICIENCY'
                excluded_cols = {prev_name_col.upper(), curr_name_col.upper(), "_TEMP_NAME_CLEANED", "STATUS", "PROFICIENCY/DEFICIENCY"}
                all_subject_cols = sorted(list(set(prev_df.columns) | set(curr_df.columns) - excluded_cols))

                subjects_data = []
                for subj in all_subject_cols:
                    prev_grade = pd.to_numeric(row_prev.get(subj, pd.NA), errors="coerce")
                    curr_grade = pd.to_numeric(row_curr.get(subj, pd.NA), errors="coerce")

                    # Assuming an overall P/D column named "PROFICIENCY/DEFICIENCY" in acad_hist_map sheets
                    proficiency_deficiency_overall = str(row_curr.get("PROFICIENCY/DEFICIENCY", ""))


                    subjects_data.append({
                        "Subject": subj,
                        "Previous Grade": prev_grade,
                        "Current Grade": curr_grade,
                        "Subject_Column": subj # Store original column name for update
                    })

                df_acad = pd.DataFrame(subjects_data)

                # Calculate Increase/Decrease and Status (for display only)
                df_acad["Increase/Decrease"] = df_acad["Current Grade"] - df_acad["Previous Grade"]
                df_acad["Increase/Decrease"] = df_acad["Increase/Decrease"].apply(
                    lambda x: "⬆️" if pd.notna(x) and x > 0 else ("⬇️" if pd.notna(x) and x < 0 else "➡️" if pd.notna(x) else "")
                )
                df_acad["Status"] = df_acad["Current Grade"].apply(
                    lambda x: "PROFICIENT" if pd.notna(x) and x >= 7 else ("DEFICIENT" if pd.notna(x) else "N/A")
                )

                # Add a column for user input for Proficiency/Deficiency for each subject
                df_acad["Proficiency/Deficiency"] = proficiency_deficiency_overall

                st.subheader("📝 Editable Grades Table")

                # Define column configuration for st.data_editor
                column_configuration_acad = {
                    "Subject": st.column_config.Column(
                        "Subject",
                        help="Academic Subject",
                        width="medium",
                        disabled=True
                    ),
                    "Previous Grade": st.column_config.NumberColumn(
                        "Previous Grade",
                        help="Previous Term Grade",
                        format="%0.2f",
                        min_value=0.0,
                        max_value=10.0, # Assuming max grade of 10
                        step=0.01,
                    ),
                    "Current Grade": st.column_config.NumberColumn(
                        "Current Grade",
                        help="Current Term Grade",
                        format="%0.2f",
                        min_value=0.0,
                        max_value=10.0, # Assuming max grade of 10
                        step=0.01,
                    ),
                    "Increase/Decrease": st.column_config.Column(
                        "Increase/Decrease",
                        help="Change from Previous to Current Grade",
                        width="small",
                        disabled=True
                    ),
                    "Status": st.column_config.Column(
                        "Status",
                        help="Proficiency Status",
                        width="small",
                        disabled=True,
                        css_class=get_status_css_class # Using CSS class for styling
                    ),
                    "Proficiency/Deficiency": st.column_config.TextColumn(
                        "Proficiency/Deficiency",
                        help="Overall Proficiency/Deficiency notes for Academics",
                        width="large",
                        max_chars=500
                    ),
                    "Subject_Column": st.column_config.Column(
                        "Subject_Column",
                        help="Original subject column name for update (hidden)",
                        disabled=True,
                        width="column_config.Column.HIDDEN" # Hide this column
                    )
                }

                edited_df = st.data_editor(
                    df_acad,
                    column_config=column_configuration_acad,
                    hide_index=True,
                    num_rows="dynamic",
                    key="acad_data_editor"
                )

                # Re-calculate status and increase/decrease after edits for display purposes
                # This ensures the status and arrow update visually as user types
                edited_df["Increase/Decrease"] = edited_df["Current Grade"] - edited_df["Previous Grade"]
                edited_df["Increase/Decrease"] = edited_df["Increase/Decrease"].apply(
                    lambda x: "⬆️" if pd.notna(x) and x > 0 else ("⬇️" if pd.notna(x) and x < 0 else "➡️" if pd.notna(x) else "")
                )
                edited_df["Status"] = edited_df["Current Grade"].apply(
                    lambda x: "PROFICIENT" if pd.notna(x) and x >= 7 else ("DEFICIENT" if pd.notna(x) else "N/A")
                )

                # Check if grades or proficiency/deficiency actually changed
                acad_changes_detected = False
                # If the overall Proficiency/Deficiency (which is propagated to all rows) has changed
                if not edited_df.empty and (edited_df.iloc[0]["Proficiency/Deficiency"] != proficiency_deficiency_overall):
                    acad_changes_detected = True
                else: # Check if any grade changed
                    for idx, row in edited_df.iterrows():
                        original_row = df_acad.loc[idx]
                        if pd.notna(row['Previous Grade']) and pd.notna(original_row['Previous Grade']) and row['Previous Grade'] != original_row['Previous Grade']:
                            acad_changes_detected = True
                            break
                        if pd.notna(row['Current Grade']) and pd.notna(original_row['Current Grade']) and row['Current Grade'] != original_row['Current Grade']:
                            acad_changes_detected = True
                            break

                if acad_changes_detected or st.session_state.get("force_show_acad_submit", False):
                    st.success("✅ Detected changes. Click below to apply updates.")
                    if st.button("📤 Submit All Academic Changes", key="submit_acad_changes"):
                        st.session_state["force_show_acad_submit"] = False
                        try:
                            # 1. Update Current Grades (History) sheet
                            # Find the row index for the cadet in the full curr_df
                            cadet_curr_row_idx = curr_df[curr_df["_TEMP_NAME_CLEANED"] == name_clean].index

                            if cadet_curr_row_idx.empty:
                                # Cadet not found, create a new row
                                new_cadet_row = {col: "" for col in curr_df.columns}
                                new_cadet_row[curr_name_col] = name_disp # Use display name for new entry
                                curr_df = pd.concat([curr_df, pd.DataFrame([new_cadet_row])], ignore_index=True)
                                cadet_curr_row_idx = curr_df.index[-1:] # Get index of newly added row
                                st.toast(f"Created new Academic History record for {name_disp}.")

                            # Apply edited values to the current DataFrame
                            for _, r in edited_df.iterrows():
                                subject_col_name = r["Subject_Column"] # Use the original subject column name

                                # Ensure subject column exists in the main DataFrame before assigning
                                if subject_col_name not in curr_df.columns:
                                    curr_df[subject_col_name] = ""
                                curr_df.loc[cadet_curr_row_idx, subject_col_name] = str(r["Current Grade"]) if pd.notna(r["Current Grade"]) else ""

                            # Update Proficiency/Deficiency column
                            # Assuming a single P/D column per cadet in acad history sheet for now
                            if "PROFICIENCY/DEFICIENCY" not in curr_df.columns:
                                curr_df["PROFICIENCY/DEFICIENCY"] = ""
                            if not edited_df.empty:
                                # Take the P/D from the first row of the edited df, assuming it's uniform
                                curr_df.loc[cadet_curr_row_idx, "PROFICIENCY/DEFICIENCY"] = edited_df.iloc[0]["Proficiency/Deficiency"]


                            # Drop the temporary cleaned name column before saving
                            curr_df_to_save = curr_df.drop(columns=["_TEMP_NAME_CLEANED"], errors='ignore')
                            update_google_sheet(curr_sheet_name, curr_df_to_save)


                            # 2. Update Previous Grades sheet (if they were edited)
                            cadet_prev_row_idx = prev_df[prev_df["_TEMP_NAME_CLEANED"] == name_clean].index
                            if not cadet_prev_row_idx.empty:
                                for _, r in edited_df.iterrows():
                                    subject_col_name = r["Subject_Column"]
                                    # Only update if previous grade was actually changed in the grid
                                    if pd.notna(r["Previous Grade"]) and \
                                       pd.to_numeric(prev_df.loc[cadet_prev_row_idx, subject_col_name].iloc[0], errors='coerce') != r["Previous Grade"]:
                                        if subject_col_name not in prev_df.columns:
                                            prev_df[subject_col_name] = "" # Add new subject column if it doesn't exist
                                        prev_df.loc[cadet_prev_row_idx, subject_col_name] = str(r["Previous Grade"]) if pd.notna(r["Previous Grade"]) else ""
                                        st.toast(f"Updated PREVIOUS GRADE for {r['Subject']} in {prev_sheet_name}")

                                prev_df_to_save = prev_df.drop(columns=["_TEMP_NAME_CLEANED"], errors='ignore')
                                update_google_sheet(prev_sheet_name, prev_df_to_save)
                            else:
                                st.warning(f"Cadet {name_disp} not found in '{prev_sheet_name}' for previous grade update.")

                            st.session_state["active_tab"] = "t2" # Keep Academics tab active
                            st.rerun() # Re-render to show updated data and clear success message
                        except Exception as e:
                            st.error(f"❌ Error saving academic changes: {e}")
                else:
                    st.session_state["force_show_acad_submit"] = True
                    st.info("📝 No detected grade or proficiency/deficiency changes yet. Try editing a cell.")

            except Exception as e:
                st.error(f"❌ Unexpected academic error: {e}")

        # --- Tab 3: PFT ---
        with t3:
            pft_sheet_map = {
                "1CL": {"1st Term": "1CL PFT", "2nd Term": "1CL PFT 2"},
                "2CL": {"1st Term": "2CL PFT", "2nd Term": "2CL PFT 2"},
                "3CL": {"1st Term": "3CL PFT", "2nd Term": "3CL PFT 2"}
            }

            if "selected_pft_term" not in st.session_state:
                st.session_state.selected_pft_term = "1st Term"

            term = st.radio(
                "Select Term for PFT",
                ["1st Term", "2nd Term"],
                index=["1st Term", "2nd Term"].index(st.session_state.selected_pft_term),
                horizontal=True,
                key="pft_term_radio"
            )
            st.session_state.selected_pft_term = term

            current_pft_sheet_name = pft_sheet_map.get(cls, {}).get(term)

            if not current_pft_sheet_name:
                st.warning(f"No PFT sheet mapped for class {cls} and term {term}.")
            else:
                pft_df_raw = sheet_df(current_pft_sheet_name)
                if pft_df_raw.empty:
                    st.info(f"No PFT data found in '{current_pft_sheet_name}'.")
                    # If empty, create a dummy one to allow new entries
                    pft_df_raw = pd.DataFrame(columns=['NAME'])

                pft_df = clean_column_names(pft_df_raw.copy())
                pft_name_col = find_name_column(pft_df.columns)

                if pft_name_col is None:
                    pft_df.insert(0, "NAME", "")
                    pft_name_col = "NAME"

                pft_df["_TEMP_NAME_CLEANED"] = pft_df[pft_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                cadet_pft_data = pft_df[pft_df["_TEMP_NAME_CLEANED"] == name_clean].copy()

                if cadet_pft_data.empty:
                    st.warning(f"No PFT record found for {name_disp} in '{current_pft_sheet_name}'. A new row will be created upon submission.")
                    # Create a blank row for the cadet if not found
                    new_row_pft = {col: "" for col in pft_df.columns}
                    new_row_pft[pft_name_col] = name_disp
                    # Initialize standard PFT columns if they don't exist
                    for col in ["PUSHUPS", "SITUPS", "PULLUPS/FLEXARM", "RUN", "PUSHUPS_GRADES", "SITUPS_GRADES", "PULLUPS_GRADES", "RUN_GRADES", "PROFICIENCY/DEFICIENCY"]:
                        if col.upper() not in [c.upper() for c in new_row_pft.keys()]: # Check case-insensitively
                            new_row_pft[col.upper()] = "" # Use upper for consistency
                    cadet_pft_data = pd.DataFrame([new_row_pft])
                    cadet_pft_data["_TEMP_NAME_CLEANED"] = name_clean # Add for internal use


                # Prepare data for st.data_editor
                pft_records = []
                exercises_config = [
                    ("Pushups", "PUSHUPS", "PUSHUPS_GRADES"),
                    ("Situps", "SITUPS", "SITUPS_GRADES"),
                    ("Pullups/Flexarm", "PULLUPS/FLEXARM", "PULLUPS_GRADES"),
                    ("3.2KM Run", "RUN", "RUN_GRADES")
                ]

                # Retrieve the overall P/D for PFT if it exists
                pft_proficiency_deficiency = str(cadet_pft_data.iloc[0].get("PROFICIENCY/DEFICIENCY", ""))


                for label, raw_col, grade_col in exercises_config:
                    reps = pd.to_numeric(cadet_pft_data.iloc[0].get(raw_col, pd.NA), errors='coerce')
                    grade = pd.to_numeric(cadet_pft_data.iloc[0].get(grade_col, pd.NA), errors='coerce')
                    status = evaluate_status(grade)
                    pft_records.append({
                        "Exercise": label,
                        "Repetitions": reps,
                        "Grade": grade,
                        "Rep_Column": raw_col, # Store original column names for update
                        "Grade_Column": grade_col,
                        "Status": status
                    })
                pft_display_df = pd.DataFrame(pft_records)

                st.subheader(f"🏋️ PFT Data – {term}")

                column_configuration_pft = {
                    "Exercise": st.column_config.Column(
                        "Exercise",
                        help="PFT Exercise",
                        width="medium",
                        disabled=True
                    ),
                    "Repetitions": st.column_config.NumberColumn(
                        "Repetitions",
                        help="Number of Repetitions/Time",
                        format="%d",
                        min_value=0,
                        step=1,
                    ),
                    "Grade": st.column_config.NumberColumn(
                        "Grade",
                        help="Score/Grade for Exercise",
                        format="%0.2f",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                    ),
                    "Status": st.column_config.Column(
                        "Status",
                        help="Proficiency Status",
                        width="small",
                        disabled=True,
                        css_class=get_status_css_class # Using CSS class for styling
                    ),
                    "Rep_Column": st.column_config.Column(
                        "Rep_Column",
                        disabled=True,
                        width="column_config.Column.HIDDEN"
                    ),
                    "Grade_Column": st.column_config.Column(
                        "Grade_Column",
                        disabled=True,
                        width="column_config.Column.HIDDEN"
                    )
                }

                edited_pft_df = st.data_editor(
                    pft_display_df,
                    column_config=column_configuration_pft,
                    hide_index=True,
                    num_rows="dynamic",
                    key="pft_data_editor"
                )

                # Re-calculate status after edits for display purposes
                edited_pft_df["Status"] = edited_pft_df["Grade"].apply(
                    lambda x: "PROFICIENT" if pd.notna(x) and x >= 7 else ("DEFICIENT" if pd.notna(x) else "N/A")
                )

                # Add proficiency/deficiency input
                edited_pft_proficiency_deficiency = st.text_area(
                    "Proficiency/Deficiency for PFT (Overall)",
                    value=pft_proficiency_deficiency,
                    key="pft_pd_text_area",
                    height=70
                )

                # Check for changes in Repetitions, Grade, or Proficiency/Deficiency
                pft_changes_detected = False
                if edited_pft_proficiency_deficiency != pft_proficiency_deficiency:
                    pft_changes_detected = True
                else:
                    for idx, row in edited_pft_df.iterrows():
                        original_row = pft_display_df.loc[idx]
                        if pd.notna(row['Repetitions']) and pd.notna(original_row['Repetitions']) and row['Repetitions'] != original_row['Repetitions']:
                            pft_changes_detected = True
                            break
                        if pd.notna(row['Grade']) and pd.notna(original_row['Grade']) and row['Grade'] != original_row['Grade']:
                            pft_changes_detected = True
                            break

                if pft_changes_detected or st.session_state.get("force_show_pft_submit", False):
                    st.success("✅ Detected changes in PFT data. Click below to apply updates.")
                    if st.button(f"📤 Submit PFT Changes ({term})", key="submit_pft_changes"):
                        st.session_state["force_show_pft_submit"] = False

                        # Find the row index for the cadet in the full pft_df
                        cadet_row_idx_in_full_df = pft_df[pft_df["_TEMP_NAME_CLEANED"] == name_clean].index

                        if cadet_row_idx_in_full_df.empty:
                            # Cadet not found, create a new row
                            new_cadet_row = {col: "" for col in pft_df.columns}
                            new_cadet_row[pft_name_col] = name_disp # Use display name for new entry
                            pft_df = pd.concat([pft_df, pd.DataFrame([new_cadet_row])], ignore_index=True)
                            cadet_row_idx_in_full_df = pft_df.index[-1:] # Get index of newly added row
                            st.toast(f"Created new PFT record for {name_disp}.")

                        # Apply edited values to the full DataFrame
                        for _, r in edited_pft_df.iterrows():
                            reps_col = r["Rep_Column"]
                            grade_col = r["Grade_Column"]

                            # Ensure columns exist in the main DataFrame before assigning
                            if reps_col.upper() not in [c.upper() for c in pft_df.columns]:
                                pft_df[reps_col.upper()] = ""
                            if grade_col.upper() not in [c.upper() for c in pft_df.columns]:
                                pft_df[grade_col.upper()] = ""

                            pft_df.loc[cadet_row_idx_in_full_df, reps_col.upper()] = str(r["Repetitions"]) if pd.notna(r["Repetitions"]) else ""
                            pft_df.loc[cadet_row_idx_in_full_df, grade_col.upper()] = str(r["Grade"]) if pd.notna(r["Grade"]) else ""

                        # Update proficiency/deficiency
                        if "PROFICIENCY/DEFICIENCY" not in pft_df.columns:
                            pft_df["PROFICIENCY/DEFICIENCY"] = ""
                        pft_df.loc[cadet_row_idx_in_full_df, "PROFICIENCY/DEFICIENCY"] = edited_pft_proficiency_deficiency


                        # Drop the temporary cleaned name column before saving
                        pft_df_to_save = pft_df.drop(columns=["_TEMP_NAME_CLEANED"], errors='ignore')
                        update_google_sheet(current_pft_sheet_name, pft_df_to_save)
                        st.session_state["active_tab"] = "t3" # Keep PFT tab active
                        st.rerun()
                else:
                    st.session_state["force_show_pft_submit"] = True
                    st.info("📝 No detected PFT changes yet. Try editing a cell.")

        # --- Tab 4: Military ---
        with t4:
            mil_sheet_map = {
                "1CL": {"1st Term": "1CL MIL", "2nd Term": "1CL MIL 2"},
                "2CL": {"1st Term": "2CL MIL", "2nd Term": "2CL MIL 2"},
                "3CL": {"1st Term": "3CL MIL", "2nd Term": "3CL MIL 2"}
            }

            if "selected_mil_term" not in st.session_state:
                st.session_state.selected_mil_term = "1st Term"

            term = st.radio(
                "Select Term for Military",
                ["1st Term", "2nd Term"],
                index=["1st Term", "2nd Term"].index(st.session_state.selected_mil_term),
                horizontal=True,
                key="mil_term_radio"
            )
            st.session_state.selected_mil_term = term

            current_mil_sheet_name = mil_sheet_map.get(cls, {}).get(term)

            if not current_mil_sheet_name:
                st.warning(f"No Military sheet mapped for class {cls} and term {term}.")
            else:
                mil_df_raw = sheet_df(current_mil_sheet_name)
                if mil_df_raw.empty:
                    st.info(f"No Military data found in '{current_mil_sheet_name}'.")
                    mil_df_raw = pd.DataFrame(columns=['NAME'])

                mil_df = clean_column_names(mil_df_raw.copy())
                mil_name_col = find_name_column(mil_df.columns)

                if mil_name_col is None:
                    mil_df.insert(0, "NAME", "")
                    mil_name_col = "NAME"

                mil_df["_TEMP_NAME_CLEANED"] = mil_df[mil_name_col].astype(str).apply(clean_cadet_name_for_comparison)
                cadet_mil_data = mil_df[mil_df["_TEMP_NAME_CLEANED"] == name_clean].copy()

                if cadet_mil_data.empty:
                    st.warning(f"No Military record found for {name_disp} in '{current_mil_sheet_name}'. A new row will be created upon submission.")
                    # Create a blank row for the cadet if not found
                    new_row_mil = {col: "" for col in mil_df.columns}
                    new_row_mil[mil_name_col] = name_disp
                    for col in ["GRADE", "AS", "NS", "AFS", "MS231", "PROFICIENCY/DEFICIENCY"]: # Ensure these exist
                        if col.upper() not in [c.upper() for c in new_row_mil.keys()]:
                            new_row_mil[col.upper()] = "" # Use upper for consistency
                    cadet_mil_data = pd.DataFrame([new_row_mil])
                    cadet_mil_data["_TEMP_NAME_CLEANED"] = name_clean

                mil_grades_list = []
                mil_columns_to_edit = []

                mil_proficiency_deficiency = str(cadet_mil_data.iloc[0].get("PROFICIENCY/DEFICIENCY", ""))

                if cls == "1CL":
                    mil_columns_to_edit = [("GRADE", "Overall Grade")] # Assuming 'GRADE' is for 1CL overall
                    for col_name, display_name in mil_columns_to_edit:
                        grade_val = pd.to_numeric(cadet_mil_data.iloc[0].get(col_name, pd.NA), errors='coerce')
                        mil_grades_list.append({"Metric": display_name, "Grade": grade_val, "Column_Name": col_name, "Status": evaluate_status(grade_val)})
                elif cls == "2CL":
                    mil_columns_to_edit = [("AS", "AS Grade"), ("NS", "NS Grade"), ("AFS", "AFS Grade")]
                    for col_name, display_name in mil_columns_to_edit:
                        grade_val = pd.to_numeric(cadet_mil_data.iloc[0].get(col_name, pd.NA), errors='coerce')
                        mil_grades_list.append({"Metric": display_name, "Grade": grade_val, "Column_Name": col_name, "Status": evaluate_status(grade_val)})
                elif cls == "3CL":
                    mil_columns_to_edit = [("MS231", "MS231 Grade")]
                    for col_name, display_name in mil_columns_to_edit:
                        grade_val = pd.to_numeric(cadet_mil_data.iloc[0].get(col_name, pd.NA), errors='coerce')
                        mil_grades_list.append({"Metric": display_name, "Grade": grade_val, "Column_Name": col_name, "Status": evaluate_status(grade_val)})

                mil_display_df = pd.DataFrame(mil_grades_list)

                st.subheader(f"📋 Military Grades – {term}")

                column_configuration_mil = {
                    "Metric": st.column_config.Column(
                        "Metric",
                        help="Military Metric",
                        width="medium",
                        disabled=True
                    ),
                    "Grade": st.column_config.NumberColumn(
                        "Grade",
                        help="Score/Grade for Military Metric",
                        format="%0.2f",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.01,
                    ),
                    "Status": st.column_config.Column(
                        "Status",
                        help="Proficiency Status",
                        width="small",
                        disabled=True,
                        css_class=get_status_css_class # Using CSS class for styling
                    ),
                    "Column_Name": st.column_config.Column(
                        "Column_Name",
                        disabled=True,
                        width="column_config.Column.HIDDEN"
                    )
                }

                edited_mil_df = st.data_editor(
                    mil_display_df,
                    column_config=column_configuration_mil,
                    hide_index=True,
                    num_rows="dynamic",
                    key="mil_data_editor"
                )

                # Re-calculate status after edits for display purposes
                edited_mil_df["Status"] = edited_mil_df["Grade"].apply(
                    lambda x: "PROFICIENT" if pd.notna(x) and x >= 7 else ("DEFICIENT" if pd.notna(x) else "N/A")
                )

                # Add proficiency/deficiency input
                edited_mil_proficiency_deficiency = st.text_area(
                    "Proficiency/Deficiency for Military (Overall)",
                    value=mil_proficiency_deficiency,
                    key="mil_pd_text_area",
                    height=70
                )

                mil_changes_detected = False
                if edited_mil_proficiency_deficiency != mil_proficiency_deficiency:
                    mil_changes_detected = True
                else:
                    for idx, row in edited_mil_df.iterrows():
                        original_row = mil_display_df.loc[idx]
                        if pd.notna(row['Grade']) and pd.notna(original_row['Grade']) and row['Grade'] != original_row['Grade']:
                            mil_changes_detected = True
                            break

                if mil_changes_detected or st.session_state.get("force_show_mil_submit", False):
                    st.success("✅ Detected changes in Military data. Click below to apply updates.")
                    if st.button(f"📤 Submit Military Changes ({term})", key="submit_mil_changes"):
                        st.session_state["force_show_mil_submit"] = False

                        cadet_row_idx_in_full_df = mil_df[mil_df["_TEMP_NAME_CLEANED"] == name_clean].index

                        if cadet_row_idx_in_full_df.empty:
                            new_cadet_row = {col: "" for col in mil_df.columns}
                            new_cadet_row[mil_name_col] = name_disp
                            mil_df = pd.concat([mil_df, pd.DataFrame([new_cadet_row])], ignore_index=True)
                            cadet_row_idx_in_full_df = mil_df.index[-1:]
                            st.toast(f"Created new Military record for {name_disp}.")


                        for _, r in edited_mil_df.iterrows():
                            target_col = r["Column_Name"]
                            if target_col.upper() not in [c.upper() for c in mil_df.columns]:
                                mil_df[target_col.upper()] = "" # Add column if it doesn't exist
                            mil_df.loc[cadet_row_idx_in_full_df, target_col.upper()] = str(r["Grade"]) if pd.notna(r["Grade"]) else ""

                        # Update proficiency/deficiency
                        if "PROFICIENCY/DEFICIENCY" not in mil_df.columns:
                            mil_df["PROFICIENCY/DEFICIENCY"] = ""
                        mil_df.loc[cadet_row_idx_in_full_df, "PROFICIENCY/DEFICIENCY"] = edited_mil_proficiency_deficiency

                        mil_df_to_save = mil_df.drop(columns=["_TEMP_NAME_CLEANED"], errors='ignore')
                        update_google_sheet(current_mil_sheet_name, mil_df_to_save)
                        st.session_state["active_tab"] = "t4" # Keep Military tab active
                        st.rerun()
                else:
                    st.session_state["force_show_mil_submit"] = True
                    st.info("📝 No detected Military changes yet. Try editing a cell.")
