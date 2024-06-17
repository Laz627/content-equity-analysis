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
    if score > high_thresh:
        return "High"
    elif score > med_thresh:
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
                df_normalized[col] = 0
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

       columns_to_use = st.multiselect(
            "Select columns to use in analysis (unselected columns will be omitted):",
            options=list(weights_mapping.keys()),
            default=list(weights_mapping.keys())
        )
    
    uploaded_file = st.file_uploader("Upload your XLSX file", type="xlsx")

    weights_mapping = {
        "Inlinks": 4, "backlinks": 7, "referring_domains_score": 10, "trust_flow_score": 8,
        "citation_flow_score": 4, "gsc_clicks_score": 14, "gsc_impressions_score": 8,
        "unique_pageviews_organic_score": 6, "unique_pageviews_all_traffic_score": 6,
        "completed_goals_all_traffic_score": 6, "number_of_keywords_page_1_score": 10,
        "number_of_keywords_page_2_score": 8, "number_of_keywords_page_3_score": 6,
        "total_search_volume_score": 5
    }

    if uploaded_file is not None:
        with st.spinner('Processing your file...'):
            try:
                equity_data_df = pd.read_excel(uploaded_file, sheet_name='Equity Data')
                keyword_data_df = pd.read_excel(uploaded_file, sheet_name='Keyword Data')
            except Exception as e:
                st.error(f"Error reading sheets: {e}")
                return

            equity_data_df.columns = [col.strip().lower().replace(' ', '_') for col in equity_data_df.columns]
            keyword_data_df.columns = [col.strip().lower().replace(' ', '_') for col in keyword_data_df.columns]

            # Fill N/A values with 0 in both dataframes
            equity_data_df = equity_data_df.fillna(0)
            keyword_data_df = keyword_data_df.fillna(0)

            # Ensure 'citation_flow_score' has no zero values to avoid division errors
            equity_data_df['citation_flow_score'] = np.where(equity_data_df['citation_flow_score'] == 0, 0.01, equity_data_df['citation_flow_score'])

            analysis_cols_to_numeric = columns_to_use + ["total_search_volume_score", "number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
            for col in analysis_cols_to_numeric:
                if col in equity_data_df.columns:
                    equity_data_df[col] = equity_data_df[col].apply(convert_to_numeric)

            keyword_data_df['ranking_position'] = pd.to_numeric(keyword_data_df['ranking_position'], errors='coerce')
            keyword_data_df['search_volume'] = pd.to_numeric(keyword_data_df['search_volume'], errors='coerce')

            required_keyword_columns = ["url", "keywords", "search_volume", "ranking_position"]
            if all(col in keyword_data_df.columns for col in required_keyword_columns):
                keyword_summary_df = keyword_data_df.groupby("url").agg(
                    total_search_volume_score=("search_volume", "sum"),
                    number_of_keywords_page_1_score=("ranking_position", lambda x: (x <= 10).sum()),
                    number_of_keywords_page_2_score=("ranking_position", lambda x: ((x > 10) & (x <= 20)).sum()),
                    number_of_keywords_page_3_score=("ranking_position", lambda x: ((x > 20) & (x <= 30)).sum())
                ).reset_index()
                keyword_summary_df = keyword_summary_df.fillna(0)
            else:
                st.error("Keyword file is missing required columns: 'URL', 'Keywords', 'Search Volume', 'Ranking Position'")
                return

            if 'url' in equity_data_df.columns and keyword_summary_df is not None and 'url' in keyword_summary_df.columns:
                equity_data_df = equity_data_df.merge(keyword_summary_df, on="url", how="left")
            else:
                st.error("'url' column is missing in one of the uploaded sheets.")
                return

            equity_data_df = equity_data_df.fillna(0)

            norm_data_df = normalize(equity_data_df.copy(), columns_to_use)

            weighted_scores_sum = pd.Series(np.zeros(len(norm_data_df)), index=norm_data_df.index)
            for column in columns_to_use:
                if column in norm_data_df.columns:
                    weight = weights_mapping[column]
                    weighted_sum = norm_data_df[column] * weight
                    weighted_scores_sum += weighted_sum

            if "trust_flow_score" in columns_to_use and "citation_flow_score" in columns_to_use:
                if (norm_data_df["citation_flow_score"] != 0).all():
                    trust_ratio = (norm_data_df["trust_flow_score"] / norm_data_df["citation_flow_score"]).fillna(0) * 2
                else:
                    trust_ratio = pd.Series(np.zeros(len(norm_data_df)), index=norm_data_df.index)
                weighted_scores_sum += trust_ratio

            equity_data_df["Final_Weighted_Score"] = weighted_scores_sum

            high_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.85)
            medium_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.50)

            equity_data_df["Recommendation"] = equity_data_df["Final_Weighted_Score"].apply(
                lambda x: classify_score(x, high_threshold, medium_threshold)
            )

            action_mapping = {
                "High": "Keep or Maintain 1:1 redirect",
                "Medium": "Update or consolidate content",
                "Low": "Evaluate URL priority or deprecate",
                "No value": "Do not keep or migrate"
            }
            equity_data_df["Action"] = equity_data_df["Recommendation"].map(action_mapping)

            keyword_columns = ["total_search_volume_score", "number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
            for col in keyword_columns:
                if col in equity_data_df.columns and col not in columns_to_use:
                    columns_to_use.append(col)

            export_columns = [col for col in equity_data_df.columns if col in equity_data_df and col not in ("Final_Weighted_Score",)]
            result_df = equity_data_df[export_columns]

            result_df = result_df.fillna(0)

        st.write("Classification Distribution:")
        st.write(equity_data_df["Recommendation"].value_counts())

        st.write("Detailed URL Table:")
        st.write(result_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Results')
        st.markdown(get_excel_download_link(output, "equity_analysis_results.xlsx"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
