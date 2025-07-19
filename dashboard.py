# --- Foxtrot CIS Streamlit Modular Inline Editor Dashboard ---
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# --- Google Sheets Setup ---
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
client = gspread.authorize(creds)
SS = client.open_by_key("1hWS_Chzs33Cp0yOV5woXZEw7inHiVxVeSE2TYcrDdc8")

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🦊 Foxtrot CIS Dashboard")

# --- Helper Functions ---
def sheet_df(sheet_name):
    ws = SS.worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    return df

def save_df_to_sheet(sheet_name, df):
    try:
        worksheet = SS.worksheet(sheet_name)
        worksheet.clear()
        worksheet.update([df.columns.tolist()] + df.values.tolist())
        return True, "Saved successfully."
    except Exception as e:
        return False, f"Error saving to {sheet_name}: {e}"

def render_editable_table(df, sheet_name, edit_key, label="Data"):
    edit_mode = st.checkbox(f"✏️ Edit Mode - {label}", key=f"{edit_key}_edit")
    if edit_mode:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"{edit_key}_editor")
        if st.button("💾 Save Changes", key=f"{edit_key}_save"):
            success, msg = save_df_to_sheet(sheet_name, edited_df)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.dataframe(df, use_container_width=True)

# --- Tabs ---
t1, t2, t3, t4, t5 = st.tabs(["📊 PFT", "🎓 Academics", "🪖 Military", "📜 Conduct", "⚙️ Admin"])

# --- T1: PFT ---
with t1:
    st.header("📊 Physical Fitness Test")
    for cls in ["1CL", "2CL", "3CL"]:
        sheet_name = f"{cls} PFT"
        try:
            df = sheet_df(sheet_name)
            st.subheader(f"{cls} PFT")
            render_editable_table(df, sheet_name, f"{cls.lower()}_pft", label=cls)
        except Exception as e:
            st.warning(f"Could not load {cls} PFT: {e}")

# --- T2: Academics ---
with t2:
    st.header("🎓 Academic Grades")
    for cls in ["1CL", "2CL", "3CL"]:
        sheet_name = f"{cls} ACAD"
        try:
            df = sheet_df(sheet_name)
            st.subheader(f"{cls} Academics")
            render_editable_table(df, sheet_name, f"{cls.lower()}_acad", label=cls)
        except Exception as e:
            st.warning(f"Could not load {cls} ACAD: {e}")

# --- T3: Military ---
with t3:
    st.header("🪖 Military Standing")
    for cls in ["1CL", "2CL", "3CL"]:
        sheet_name = f"{cls} MIL"
        try:
            df = sheet_df(sheet_name)
            st.subheader(f"{cls} Military")
            render_editable_table(df, sheet_name, f"{cls.lower()}_mil", label=cls)
        except Exception as e:
            st.warning(f"Could not load {cls} MIL: {e}")

# --- T4: Conduct ---
with t4:
    st.header("📜 Conduct Reports")
    for cls in ["1CL", "2CL", "3CL"]:
        sheet_name = f"{cls} CONDUCT"
        try:
            df = sheet_df(sheet_name)
            st.subheader(f"{cls} Conduct")
            render_editable_table(df, sheet_name, f"{cls.lower()}_conduct", label=cls)
        except Exception as e:
            st.warning(f"Could not load {cls} CONDUCT: {e}")

# --- T5: Admin Tab ---
with t5:
    st.header("⚙️ Admin Tools")
    st.markdown("Use this tab to test sheet access, refresh, or add more editing utilities.")
    test_sheet = st.text_input("Test Sheet Name", value="1CL ACAD")
    if st.button("🔍 Preview Sheet"):
        try:
            test_df = sheet_df(test_sheet)
            st.dataframe(test_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading sheet: {e}")
