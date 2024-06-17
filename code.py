import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

# Check if openpyxl is installed
try:
    import openpyxl
except ImportError:
    st.error("The openpyxl library is required to handle Excel files. Please install it using: `pip install openpyxl`")
    st.stop()

@st.cache_data
def convert_to_numeric(value):
    """Converts a value to numeric format after replacing commas."""
    try:
        return float(str(value).replace(',', ''))
    except ValueError:
        return value

def classify_score(score, high_thresh, med_thresh):
    """Classify the score into High, Medium, Low, or No value based on thresholds."""
    if score >= high_thresh:
        return "High"
    elif score >= med_thresh:
        return "Medium"
    elif score > 0:
        return "Low"
    else:
        return "No value"

def get_excel_download_link(output, filename):
    """Generates a link to download the DataFrame as an Excel file."""
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href

def normalize(df, metric_cols):
    """Normalize the specified columns in the DataFrame using Min-Max normalization."""
    df_normalized = df.copy()
    for col in metric_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            else: 
                df_normalized[col] = df[col]  # if all values are the same, normalization isn't necessary
    return df_normalized

def main():
    st.title("Equity Analysis Calculator")
    st.write("**Created by: Brandon Lazovic**")

    st.header("How It Works")
    st.write("""
    The Equity Analysis Calculator helps you evaluate the equity of your URLs by considering various SEO metrics.
    It allows you to upload an XLSX file with your URL data and keywords, process the data to calculate weighted scores, classify the URLs,
    and provide actionable recommendations based on their equity scores. 
    """)

    st.header("How to Use It")
    st.write("""
    1. **Download the Template**: Click the 'Download Equity Analysis Template XLSX' button to download a template file with the required columns.
    2. **Upload Your Data**: Use the 'Upload your XLSX file' button to upload your equity analysis data and keywords in separate tabs.
    3. **Select Columns**: Choose which columns to include in the analysis from the multiselect dropdown.
    4. **View Results**: The app will display the analyzed results and provide a download link for the final output file.
    """)

    st.header("Expected Outcomes")
    st.write("""
    After processing the data, the tool will:
    - Provide a weighted score for each URL based on the selected metrics.
    - Classify URLs into High, Medium, Low, or No value categories.
    - Offer actionable recommendations for each URL based on their classification.
    """)

    # Allow users to download a template XLSX file for equity and keyword analysis
    if st.button("Download Equity Analysis Template XLSX"):
        equity_template_df = pd.DataFrame(columns=[
            "URL", "status_code", "Inlinks", "backlinks", "referring_domains_score", "trust_flow_score", 
            "citation_flow_score", "gsc_clicks_score", "gsc_impressions_score", 
            "unique_pageviews_organic_score", "unique_pageviews_all_traffic_score", "completed_goals_all_traffic_score"
        ])
        keyword_template_df = pd.DataFrame(columns=[
            "URL", "Keywords", "Search Volume", "Ranking Position"
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            equity_template_df.to_excel(writer, sheet_name='Equity Data', index=False)
            keyword_template_df.to_excel(writer, sheet_name='Keyword Data', index=False)
        st.markdown(get_excel_download_link(output, "equity_analysis_template.xlsx"), unsafe_allow_html=True)

    # Allow users to upload their XLSX file with equity and keyword data in separate tabs
    uploaded_file = st.file_uploader("Upload your XLSX file", type="xlsx")

    # Column selection before running the analysis
    weights_mapping = {
        "Inlinks": 4,
        "backlinks": 7,
        "referring_domains_score": 10,
        "trust_flow_score": 8,
        "citation_flow_score": 4,
        "gsc_clicks_score": 14,
        "gsc_impressions_score": 8,
        "unique_pageviews_organic_score": 6,
        "unique_pageviews_all_traffic_score": 6,
        "completed_goals_all_traffic_score": 6,
        "number_of_keywords_page_1_score": 10,
        "number_of_keywords_page_2_score": 8,
        "number_of_keywords_page_3_score": 6,
        "total_search_volume_score": 5
    }

    columns_to_use = st.multiselect(
        "Select columns to use in analysis (unselected columns will be omitted):",
        options=list(weights_mapping.keys()),
        default=list(weights_mapping.keys())
    )

    if uploaded_file is not None:
        try:
            # Reading data and keywords from the XLSX file
            equity_data_df = pd.read_excel(uploaded_file, sheet_name='Equity Data')
            keyword_data_df = pd.read_excel(uploaded_file, sheet_name='Keyword Data')
        except Exception as e:
            st.error(f"Error reading sheets: {e}")
            return

        # Normalize column names in equity and keyword data
        equity_data_df.columns = [col.strip().lower().replace(' ', '_') for col in equity_data_df.columns]
        keyword_data_df.columns = [col.strip().lower().replace(' ', '_') for col in keyword_data_df.columns]

        # Ensure 'ranking_position' and 'search_volume' are numeric in keyword data
        keyword_data_df['ranking_position'] = pd.to_numeric(keyword_data_df['ranking_position'], errors='coerce')
        keyword_data_df['search_volume'] = pd.to_numeric(keyword_data_df['search_volume'], errors='coerce')
        
        required_keyword_columns = ["url", "keywords", "search_volume", "ranking_position"]
        if all(col in keyword_data_df.columns for col in required_keyword_columns):
            # Process keyword data to summarize
            keyword_summary_df = keyword_data_df.groupby("url").agg(
                total_search_volume_score=("search_volume", "sum"),
                number_of_keywords_page_1_score=("ranking_position", lambda x: (x <= 10).sum()),
                number_of_keywords_page_2_score=("ranking_position", lambda x: ((x > 10) & (x <= 20)).sum()),
                number_of_keywords_page_3_score=("ranking_position", lambda x: ((x > 20) & (x <= 30)).sum())
            ).reset_index()
            st.write("Keyword Summary DataFrame Preview:", keyword_summary_df.head(10))
        else:
            st.error("Keyword file is missing required columns: 'URL', 'Keywords', 'Search Volume', 'Ranking Position'")

        if 'url' in equity_data_df.columns and keyword_summary_df is not None and 'url' in keyword_summary_df.columns:
            # Merge keyword summary data with equity data
            equity_data_df = equity_data_df.merge(keyword_summary_df, on="url", how="left")
            st.write("Merged DataFrame Preview:", equity_data_df.head(10))
        else:
            st.error("'url' column is missing in one of the uploaded sheets.")

        # Correct data formatting for columns with numeric values
        columns_to_correct = [
            "total_search_volume_score",
            "number_of_keywords_page_1_score",
            "number_of_keywords_page_2_score",
            "number_of_keywords_page_3_score"
        ]

        for col in columns_to_correct:
            if col in equity_data_df.columns:
                equity_data_df[col] = equity_data_df[col].apply(convert_to_numeric)

        # Normalize the data before weighting
        norm_data_df = normalize(equity_data_df.copy(), columns_to_use)

        # Calculating the weighted scores for each metric in the adjusted dataset
        weighted_scores = []
        for column in columns_to_use:
            if column in norm_data_df.columns:
                weight = weights_mapping[column]
                weighted_scores.append(norm_data_df[column] * weight)

        # Calculate Trust Ratio Score as trust_flow_score / citation_flow_score
        if "trust_flow_score" in columns_to_use and "citation_flow_score" in columns_to_use:
            trust_ratio_weighted = (norm_data_df["trust_flow_score"] / norm_data_df["citation_flow_score"]).fillna(0) * 2
            weighted_scores.append(trust_ratio_weighted)

        # Compute the final weighted score for each URL
        equity_data_df["Final_Weighted_Score"] = np.sum(weighted_scores, axis=0)

        # Compute thresholds for classification
        high_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.85)
        medium_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.50)

        # Apply classification
        equity_data_df["Recommendation"] = equity_data_df["Final_Weighted_Score"].apply(lambda x: classify_score(x, high_threshold, medium_threshold))

        # Map the "Recommendation" to the corresponding action
        action_mapping = {
            "High": "Keep or Maintain 1:1 redirect",
            "Medium": "Update or consolidate content",
            "Low": "Evaluate URL priority or deprecate",
            "No value": "Do not keep or migrate"
        }
        equity_data_df["Action"] = equity_data_df["Recommendation"].map(action_mapping)

        # Ensure keyword columns are included in the output
        keyword_columns = ["total_search_volume_score", "number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
        for col in keyword_columns:
            if col in equity_data_df.columns and col not in columns_to_use:
                columns_to_use.append(col)
        
        # Columns to include in the final output
        export_columns = [col for col in equity_data_df.columns if col not in ("Final_Weighted_Score",)]
        result_df = equity_data_df[export_columns]

        # Show the results
        st.write(result_df)

        # Create a download link for the resulting DataFrame
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Results')
        st.markdown(get_excel_download_link(output, "equity_analysis_results.xlsx"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
