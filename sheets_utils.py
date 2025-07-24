import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# Authenticate using Streamlit secrets
def get_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)

# Get sheet data as DataFrame
def sheet_df(sheet_name):
    client = get_client()
    sheet = client.open("FoxtrotCIS")  # Change to your Google Sheet name
    ws = sheet.worksheet(sheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)

# Update whole sheet with new DataFrame
def update_sheet(sheet_name, df):
    client = get_client()
    sheet = client.open("FoxtrotCIS")
    ws = sheet.worksheet(sheet_name)
    ws.update([df.columns.values.tolist()] + df.values.tolist())

def sheet_df(sheet_name):
    ws = SS.worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    return df

def get_worksheet_by_name(name):
    for ws in SS.worksheets():
        if ws.title.strip().upper() == name.strip().upper():
            return ws
    raise Exception(f"Worksheet '{name}' not found.")

def find_name_column(df, possible_name_cols=["NAME", "FULL NAME", "CADET NAME"]):
    upper_cols = df.columns.str.upper()
    for col in possible_name_cols:
        if col.upper() in upper_cols:
            return df.columns[upper_cols == col.upper()][0]
    return None

