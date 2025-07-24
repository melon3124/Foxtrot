# sheets_utils.py

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Create authorized gspread client from Streamlit secrets
def get_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ],
    )
    return gspread.authorize(creds)

# Read a worksheet into a DataFrame
def sheet_df(sheet_name):
    client = get_client()
    sheet = client.open("FOXTROT DASHBOARD V2")  # Replace with your actual Google Sheet name
    ws = sheet.worksheet(sheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)

# Optional: update an entire worksheet from a DataFrame
def update_sheet(sheet_name, df):
    client = get_client()
    sheet = client.open("FOXTROT DASHBOARD V2")
    ws = sheet.worksheet(sheet_name)
    ws.update([df.columns.values.tolist()] + df.values.tolist())
