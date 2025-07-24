import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# Authenticate using secrets
def get_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ],
    )
    return gspread.authorize(creds)

# Get a worksheet as a DataFrame
def sheet_df(sheet_name):
    client = get_client()
    sheet = client.open("FoxtrotCIS")  # Replace with your actual sheet name
    ws = sheet.worksheet(sheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)
