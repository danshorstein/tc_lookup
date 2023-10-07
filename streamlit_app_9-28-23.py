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
        truncated_drs = [truncated_gl_code(str(dr), i) if dr else '' for dr in drs]
        truncated_crs = [truncated_gl_code(str(cr), i) if cr else '' for cr in crs]
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

# def process_and_analyze_uploaded_csv(uploaded_file, df_tc_lookup):
#     """
#     Process and analyze the uploaded CSV with advanced cancellation logic and handle cases where all SGLs are canceled out.
#     """
#     # Check if the analysis has already been done and stored in session state
#     if 'analyzed_data' in st.session_state:
#         return st.session_state['analyzed_data']
    
#     # Read the uploaded CSV into a DataFrame
#     df_uploaded = pd.read_csv(uploaded_file)
    
#     # Truncate each debit and credit amount to the first 6 digits
#     for col in df_uploaded.columns:
#         if col.startswith(('Debit', 'Credit')):
#             df_uploaded[col] = df_uploaded[col].apply(lambda x: str(x)[:6] if pd.notnull(x) else '')
    
#     # Create new columns to store results
#     df_uploaded['Matching TCs'] = None
#     df_uploaded['Match Type'] = None
    
#     # For each record, perform a TC Lookup analysis
#     for idx, row in df_uploaded.iterrows():
#         drs = [row['Debit1'], row['Debit2'], row['Debit3']]
#         crs = [row['Credit1'], row['Credit2'], row['Credit3']]
        
#         # Filter out None values
#         drs = [dr for dr in drs if dr is not None]
#         crs = [cr for cr in crs if cr is not None]
        
#         # Identify SGLs that are both in debits and credits
#         common_sglas = set(drs) & set(crs)
        
#         # For each common SGL, cancel out equal number of debits and credits
#         for sgl in common_sglas:
#             dr_count = drs.count(sgl)
#             cr_count = crs.count(sgl)
#             cancel_count = min(dr_count, cr_count)
            
#             # Remove the canceled SGLs from drs and crs
#             for _ in range(cancel_count):
#                 drs.remove(sgl)
#                 crs.remove(sgl)
        
#         # If both debit and credit lists are empty after cancellation, set it to "No Matches"
#         if not drs and not crs:
#             df_uploaded.at[idx, 'Matching TCs'] = "No Matches"
#             df_uploaded.at[idx, 'Match Type'] = "No Matches"
#             continue
        
#         # Apply filters for all the Debits and all the Credits
#         exact_matches = filter_tc_tool(df_tc_lookup, drs=drs, crs=crs)
        
#         # If there are exact matches
#         if not exact_matches.empty:
#             df_uploaded.at[idx, 'Matching TCs'] = ", ".join(exact_matches.index.to_list())
#             df_uploaded.at[idx, 'Match Type'] = "Exact Matches"
#         else:
#             # If no exact matches, then perform truncate search
#             close_matches = truncate_search(df_tc_lookup, drs, crs)
#             if not close_matches.empty:
#                 df_uploaded.at[idx, 'Matching TCs'] = ", ".join(close_matches.index.to_list())
#                 df_uploaded.at[idx, 'Match Type'] = "Close Matches"
    
#     # Store the analysis results in session state
#     st.session_state['analyzed_data'] = df_uploaded
    
#     return df_uploaded

# The function has been adjusted to consider the advanced cancellation of debits and credits to the same SGL.


def potential_concern(row, df_tc_lookup):
    matched_tcs = str(row['Matching TCs']).split(', ')
    for tc in matched_tcs:
        if tc in df_tc_lookup.index:
            tc_row = df_tc_lookup.loc[tc]
            budgetary_drs = tc_row['Budgetary_Debits']
            budgetary_crs = tc_row['Budgetary_Credits']
            proprietary_drs = tc_row['Proprietary_Debits']
            proprietary_crs = tc_row['Proprietary_Credits']

            # Check if the TC has both budgetary and proprietary debits/credits
            if (budgetary_drs or budgetary_crs) and (proprietary_drs or proprietary_crs):
                # Check if uploaded data is missing budgetary or proprietary transactions
                if any([isinstance(row[f'Debit{i}'], str) and len(row[f'Debit{i}']) > 0 and row[f'Debit{i}'][0] == '4' for i in range(1, 4)]) or \
                   any([isinstance(row[f'Credit{i}'], str) and len(row[f'Credit{i}']) > 0 and row[f'Credit{i}'][0] == '4' for i in range(1, 4)]):
                   return 'Potentially missing Proprietary'
                else:
                   return 'Potentially missing Budgetary'
    return ''



def process_and_analyze_uploaded_csv_with_categories(uploaded_file, df_tc_lookup, categories_selected):
    """
    Process and analyze the uploaded CSV considering selected TC categories and flagging potential concerns.
    """
    if 'analyzed_data' in st.session_state:
        return st.session_state['analyzed_data']
    
    df_uploaded = pd.read_csv(uploaded_file)
    
    for col in df_uploaded.columns:
        if col.startswith(('Debit', 'Credit')):
            df_uploaded[col] = df_uploaded[col].apply(lambda x: str(x)[:6] if pd.notnull(x) else '')
    
    df_uploaded['Matching TCs'] = None
    df_uploaded['Match Type'] = None
    # df_uploaded['Potential Concern'] = None
    
    for idx, row in df_uploaded.iterrows():
        drs = [row['Debit1'], row['Debit2'], row['Debit3']]
        crs = [row['Credit1'], row['Credit2'], row['Credit3']]
        drs = [dr for dr in drs if dr is not None]
        crs = [cr for cr in crs if cr is not None]
        
        common_sglas = set(drs) & set(crs)

        for sgl in common_sglas:
            if sgl in drs and sgl in crs:
                if len(drs) > len(crs):
                    drs.remove(sgl)
                else:
                    crs.remove(sgl)
        # for sgl in common_sglas:
        #     dr_count = drs.count(sgl)
        #     cr_count = crs.count(sgl)
        #     cancel_count = min(dr_count, cr_count)
        #     for _ in range(cancel_count):
        #         drs.remove(sgl)
        #         crs.remove(sgl)
        
        exact_matches = filter_tc_tool(df_tc_lookup[df_tc_lookup.index.str.startswith(tuple(categories_selected))], drs=drs, crs=crs)
        
        if not exact_matches.empty:
            df_uploaded.at[idx, 'Matching TCs'] = ", ".join(exact_matches.index.to_list())
            df_uploaded.at[idx, 'Match Type'] = "Exact Matches"
            
            # # Check for potential concern
            # for tc in exact_matches.index:
            #     bd_exists = any(exact_matches.loc[tc, col] for col in ['Budgetary_Debits', 'Budgetary_Credits'])
            #     pr_exists = any(exact_matches.loc[tc, col] for col in ['Proprietary_Debits', 'Proprietary_Credits'])
            #     if (bd_exists and not pr_exists) or (pr_exists and not bd_exists):
            #         df_uploaded.at[idx, 'Potential Concern'] = 'Yes'
            #         break
        else:
            close_matches = truncate_search(df_tc_lookup[df_tc_lookup.index.str.startswith(tuple(categories_selected))], drs, crs)
            if not close_matches.empty:
                df_uploaded.at[idx, 'Matching TCs'] = ", ".join(close_matches.index.to_list())
                df_uploaded.at[idx, 'Match Type'] = "Close Matches"
    
    st.session_state['analyzed_data'] = df_uploaded
    return df_uploaded


# def process_and_analyze_uploaded_csv_with_session_state(uploaded_file, df_tc_lookup):
#     """
#     Process and analyze the uploaded CSV using Streamlit's session state to avoid unnecessary re-analysis.
#     """
#     # Check if the analysis has already been done and stored in session state
#     if 'analyzed_data' in st.session_state:
#         return st.session_state['analyzed_data']
    
#     # If not, perform the analysis
#     df_uploaded = process_and_analyze_uploaded_csv(uploaded_file, df_tc_lookup)
    
#     # Store the analysis results in session state
#     st.session_state['analyzed_data'] = df_uploaded
    
#     return df_uploaded




def generate_excel_from_analysis(df_results, df_tc_lookup):
    """
    Generate an Excel file based on the analysis results.
    """
    towrite = io.BytesIO()
    
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        # Main tab: Resulting Analysis
        df_results.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Second tab: Filtered Trans Codes
        # Extract unique TCs from the results and filter the TC lookup table
        unique_tcs = set(tc for tcs in df_results['Matching TCs'].dropna() for tc in tcs.split(", "))
        df_filtered_tcs = df_tc_lookup[df_tc_lookup.index.isin(unique_tcs)]
        df_filtered_tcs.to_excel(writer, sheet_name='Filtered Trans Codes', index=True)
    
    towrite.seek(0)
    return towrite

def streamlit_app():
    st.title("Fed TC Lookup Tool")

    tab1, tab2 = st.tabs(['TC Analysis', 'File Analysis'])

    with tab1:

        with st.sidebar:


            # Dropdown for fiscal year selection
            fiscal_years = get_fiscal_years_from_filenames()
            selected_fy = st.selectbox("Select Fiscal Year", fiscal_years)

            # Loading data for the selected fiscal year
            df = load_tc_lookup_tool(selected_fy)

            
            # User input
            col1, col2 = st.columns(2)
            with col1:
                drs = [st.text_input(f"Debit {i+1}", value='') for i in range(3)]
            with col2:
                crs = [st.text_input(f"Credit {i+1}", value='') for i in range(3)]
        
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
                    'Criteria': ['Debits', 'Debits', 'Debits', 'Credits', 'Credits', 'Credits', 'Keyword', 'Fiscal Year'],
                    'Values': [drs[0], drs[1], drs[2], crs[0], crs[1], crs[2], keyword, selected_fy]
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

    with tab2:
        st.header("Upload CSV for Analysis")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        # Create the template DataFrame
        template_df = pd.DataFrame(columns=["ID", "Debit1", "Debit2", "Debit3", "Credit1", "Credit2", "Credit3"])

        # Convert the DataFrame to a CSV in-memory object
        csv_data = template_df.to_csv(index=False)

        # Offer the CSV for download
        st.download_button(
            label="Download CSV Template",
            data=csv_data,
            file_name="upload_template.csv",
            mime="text/csv",
        )

        # Transaction Code Categories for selection
        tc_categories = {
            'A': 'A. Funding',
            'B': 'B. Disb and Pbls',
            'C': 'C. Coll and Recvs',
            'D': 'D. Adj/Write-offs/Reclass',
            'E': 'E. Accr/Nonbudg Transfers',
            'F': 'F. Yearend',
            'G': 'G. Memo Entries',
            'H': 'H. Specialized Entries'
        }
        
        selected_values = st.multiselect('Select Trans Code Categories', list(tc_categories.values()), default=list(tc_categories.values()))

        categories_selected = [key for key, value in tc_categories.items() if value in selected_values]

        # If there's a change in the selected categories, clear the analysis results
        if 'selected_categories' not in st.session_state or set(st.session_state['selected_categories']) != set(categories_selected):
            st.session_state['selected_categories'] = categories_selected
            if 'analyzed_data' in st.session_state:
                del st.session_state['analyzed_data']

        if uploaded_file:
            # Generate a hash of the uploaded file to uniquely identify it
            file_hash = hash(uploaded_file.getvalue())
            
            # Check if the file is different from the previously uploaded file
            if 'uploaded_file_hash' not in st.session_state or st.session_state['uploaded_file_hash'] != file_hash:
                # Update the session state with the new file's hash
                st.session_state['uploaded_file_hash'] = file_hash
                
                # Clear the analysis results from the session state
                if 'analyzed_data' in st.session_state:
                    del st.session_state['analyzed_data']
                    
            fiscal_years = get_fiscal_years_from_filenames()
            selected_fy = fiscal_years[0]
            df_tc_lookup = load_tc_lookup_tool(selected_fy)
            df_results = process_and_analyze_uploaded_csv_with_categories(uploaded_file, df_tc_lookup, categories_selected)
            df_results['Potential Concern'] = df_results.apply(lambda row: potential_concern(row, df_tc_lookup), axis=1)
            
            st.subheader("Analysis Results")
            st.write(df_results)
            
            excel_data = generate_excel_from_analysis(df_results, df_tc_lookup)
            st.download_button(
                label="Download Analysis as Excel",
                data=excel_data,
                file_name="analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )



streamlit_app()
