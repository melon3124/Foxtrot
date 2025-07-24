import pandas as pd
import streamlit as st

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

