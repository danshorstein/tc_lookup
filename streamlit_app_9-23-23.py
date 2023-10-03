import io
import os

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")



# Reusing functions from the provided scripts
def load_tc_lookup_tool(fy):
    full_path = './data/'
    ussgl_tc_json_files = get_json_files(full_path)
    current_tc_tool = os.path.join(full_path, f'tc_tool_{fy}.json')
    ussgl_tc_df = load_df_from_json(current_tc_tool)
    return ussgl_tc_df

def get_json_files(full_path):
    files = [[json, os.path.join(full_path, json)] for json in list(os.walk(full_path))[0][2] if json.endswith('json')]
    return {json[0]: json[1] for json in files} if files else {}

def load_df_from_json(path):
    columns = ['Description', 'Comment', 'Reference', 'Budgetary_Debits',
               'Budgetary_Credits', 'Proprietary_Debits',
               'Proprietary_Credits', 'Memo_Debits', 'Memo_Credits']
    df = pd.read_json(path)
    df = df[columns]
    return df

def get_fiscal_years_from_filenames():
    full_path = './data/'  # Adjusting path to current directory for local run
    filenames = get_json_files(full_path).keys()
    fiscal_years = sorted([int(fname.split('_')[-1].split('.')[0]) for fname in filenames], reverse=True)
    return fiscal_years

def filter_tc_tool(df, drs=None, crs=None):
    sgls = df.copy()
    if drs:
        for dr in drs:
            if dr:
                dr = str(dr)
                sgls = sgls.where(df.Budgetary_Debits.str.contains(dr) | 
                                  df.Proprietary_Debits.str.contains(dr) |
                                  df.Memo_Debits.str.contains(dr)).dropna()
    if crs:
        for cr in crs:
            if cr:
                cr = str(cr)
                sgls = sgls.where(df.Budgetary_Credits.str.contains(cr) | 
                                  df.Proprietary_Credits.str.contains(cr) |
                                  df.Memo_Credits.str.contains(cr)).dropna()
    return sgls

def keyword_filter(df, keyword):
    return df[df['Description'].str.contains(keyword, case=False, na=False) | 
              df['Comment'].str.contains(keyword, case=False, na=False)]


def display_summary_table(df, title, return_df=False):
    """
    Display a summary table for the given dataframe.
    """
    
    # Abbreviated major categories with section names
    categories = {
        'A': 'A. Funding',
        'B': 'B. Disb and Pbls',
        'C': 'C. Coll and Recvs',
        'D': 'D. Adj/Write-offs/Reclass',
        'E': 'E. Accr/Nonbudg Transfers',
        'F': 'F. Yearend',
        'G': 'G. Memo Entries',
        'H': 'H. Specialized Entries'
    }
    
    summary_data = {}
    
    # Loop through each major category
    for code, category_name in categories.items():
        # Filter the dataframe based on the first letter of the transaction code
        subset = df[df.index.str.startswith(code)]
        
        total_records = len(subset)
        normal_records = len(subset[~subset.index.str.endswith('R')])
        reversed_records = total_records - normal_records
        
        # Update the summary data
        summary_data[category_name] = [total_records, normal_records, reversed_records]
    
    # Convert the summary data to a dataframe
    summary_df = pd.DataFrame(summary_data, index=['Total Records', 'Normal', 'Reversed']).T
    
    # Check if need to return summary DF
    if return_df:
        return summary_df

    # Display the summary table
    st.subheader(title)
    st.table(summary_df)


def truncate_search(df, drs, crs, truncate_length=3):
    """
    Attempt to find close matches by iteratively truncating the least significant digits
    of the GL codes and searching for matches.
    """
    # Helper function to perform truncation
    def truncated_gl_code(gl_code, num_chars):
        return gl_code[:-num_chars]

    # Initialize empty dataframe to store results
    matches = pd.DataFrame()

    # Iterate over the range of truncation lengths
    for i in range(1, truncate_length + 1):
        # Truncate GL codes
        truncated_drs = [truncated_gl_code(dr, i) if dr else None for dr in drs]
        truncated_crs = [truncated_gl_code(cr, i) if cr else None for cr in crs]
        # Filter using the truncated GL codes
        current_matches = filter_tc_tool(df, drs=truncated_drs, crs=truncated_crs)
        # Append results to matches dataframe
        matches = pd.concat([matches, current_matches], axis=0)

    return matches.drop_duplicates()

# Set column configuration for TC table
col_config = {
    'Description': st.column_config.Column(width='large'),
    'Comment': st.column_config.Column(width='medium'),
    'Reference': st.column_config.Column(width='medium'),
    'Budgetary_Debits': st.column_config.Column(width='small'),
    'Budgetary_Credits': st.column_config.Column(width='small'),
    'Proprietary_Debits': st.column_config.Column(width='small'),
    'Proprietary_Credits': st.column_config.Column(width='small'),
    'Memo_Debits': st.column_config.Column(width='small'),
    'Memo_Credits': st.column_config.Column(width='small'),

}

def streamlit_app():
    st.title("Fed TC Lookup Tool")

    with st.sidebar:
        # Dropdown for fiscal year selection
        fiscal_years = get_fiscal_years_from_filenames()
        selected_fy = st.selectbox("Select Fiscal Year", fiscal_years)

          # Loading data for the selected fiscal year
        df = load_tc_lookup_tool(selected_fy)

        
        # User input
        col1, col2 = st.columns(2)
        with col1:
            drs = [st.text_input(f"Debit {i+1}", value='') for i in range(2)]
        with col2:
            crs = [st.text_input(f"Credit {i+1}", value='') for i in range(2)]
      
        keyword = st.text_input("Keyword/Phrase Filter", value='')

            # Filter data based on user input
        exact_matches = filter_tc_tool(df, drs=drs, crs=crs)
        if keyword:
            exact_matches = keyword_filter(exact_matches, keyword)
            
        st.session_state['exact_matches'] = exact_matches


        # Check for close matches if no exact matches are found
        if exact_matches.empty:

            # Check for close matches using truncation-based search
            close_matches = truncate_search(df, drs, crs)
            if keyword:
                close_matches = keyword_filter(close_matches, keyword)

            st.session_state['close_matches'] = close_matches

        if 'exact_matches' not in st.session_state:
            st.session_state['exact_matches'] = pd.DataFrame()
        exact_matches = st.session_state['exact_matches']

        if 'close_matches' not in st.session_state:
            st.session_state['close_matches'] = pd.DataFrame()
        close_matches = st.session_state['close_matches']

        if not exact_matches.empty:
            display_summary_table(exact_matches, 'Filtered Results Summary')
        elif not close_matches.empty:
            display_summary_table(close_matches, 'Filtered Results Summary')
        # else:
        #     display_summary_table(df, 'Filtered Results Summary')
    



    # Display exact matches
    st.subheader("Exact Matches")


    # Link to the source PDF file
    if selected_fy == 2024:
        st.write("[FY2024 Transactions updated September 2023](https://raw.githubusercontent.com/danshorstein/tc_lookup/main/p2sec3_transactions_2024.pdf)")
    elif selected_fy == 2023:
        st.write("[FY2023 Transactions updated September 2023](https://raw.githubusercontent.com/danshorstein/tc_lookup/main/p1sec3_transactions_2023.pdf)")


    # Download as Excel functionality
    try:
        towrite = io.BytesIO()
        
        # Create an Excel writer object
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            # Sheet 1: Filtering Criteria and Summary Table
            criteria_data = {
                'Criteria': ['Debits', 'Debits', 'Credits', 'Credits', 'Keyword', 'Fiscal Year'],
                'Values': [drs[0], drs[1], crs[0], crs[1], keyword, selected_fy]
            }
            criteria_df = pd.DataFrame(criteria_data)
            criteria_df.to_excel(writer, sheet_name='Criteria & Summary', startrow=0, startcol=0, index=False)
            
            # Get the summary table as a DataFrame
            if not exact_matches.empty:
                summary_df = display_summary_table(exact_matches, 'Filtered Results Summary', return_df=True)
            elif not close_matches.empty:
                summary_df = display_summary_table(close_matches, 'Filtered Results Summary', return_df=True)
            else:
                summary_df = display_summary_table(df, 'Filtered Results Summary', return_df=True)
            
            # Add the summary table below the filtering criteria
            summary_df.to_excel(writer, sheet_name='Criteria & Summary', startrow=len(criteria_df) + 2)
            
            # Sheet 2: Resulting Trans Codes
            if not exact_matches.empty:
                exact_matches.to_excel(writer, sheet_name='Trans Codes', index=True, header=True)
            else:
                close_matches.to_excel(writer, sheet_name='Trans Codes', index=True, header=True)
        
        towrite.seek(0)
        st.download_button(
            label="Download Matches as Excel",
            data=towrite,
            file_name=f"matches_fy{selected_fy}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.write(f'EXCEPTION! {e}')


    
    if not exact_matches.empty:
        st.dataframe(exact_matches, 
                     column_config=col_config,
                     height=600)

    else:
        st.write("No exact matches found.")
        if not close_matches.empty:
            st.subheader("Close Matches")
            st.dataframe(close_matches, 
                     column_config=col_config,
                     height=600)

    



streamlit_app()
