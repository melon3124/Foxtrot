import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import unicodedata

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def init_gsheets():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["google_service_account"],
            scopes=scope
        )
        client = gspread.authorize(creds)
        return client.open("FOXTROT DASHBOARD V2")
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
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
        worksheet = init_gsheets().worksheet(name)
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
