
import streamlit as st
import pandas as pd
import os

# Reusing functions from the provided scripts
def load_tc_lookup_tool():
    full_path = './data/'  # Adjusting path to current directory for local run
    ussgl_tc_json_files = get_json_files(full_path)
    current_tc_tool = os.path.join(full_path, max(ussgl_tc_json_files))
    ussgl_tc_df = load_df_from_json(current_tc_tool)
    return ussgl_tc_df

def get_json_files(full_path):
    files = [[json, os.path.join(full_path, json)] for json in list(os.walk(full_path))[0][2] if json.endswith('json')]
    return {json[0]: json[1] for json in files} if files else {}

def load_df_from_json(path):
    columns = ['Description', 'Comment', 'Reference', 'Budgetary_Debits',
               'Budgetary_Credits', 'Proprietary_Debits',
               'Proprietary_Credits', 'Memo_Debits', 'Memo_Credits',
               'Debits', 'Credits']
    df = pd.read_json(path)
    df.columns = columns
    return df

def filter_tc_tool(df, drs=None, crs=None):
    sgls = df.copy()
    if drs:
        for dr in drs:
            dr = str(dr)
            sgls = sgls.where(df.Debits.str.contains(dr)).dropna()
    if crs:
        for cr in crs:
            cr = str(cr)
            sgls = sgls.where(df.Credits.str.contains(cr)).dropna()
    return sgls

# Main Streamlit application
def streamlit_app():
    st.title("Fed TC Tool - Fiscal Year 2024")

    # Loading data
    df = load_tc_lookup_tool()

    # User input
    dr = st.number_input("Debit", min_value=1010, max_value=999999)
    cr = st.number_input("Credit", min_value=1010, max_value=999999)

    # Filter data and display
    filtered_data = filter_tc_tool(df, drs=[dr], crs=[cr])
    st.table(filtered_data)

streamlit_app()

