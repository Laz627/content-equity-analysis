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
    """Converts a value to numeric format after replacing commas. Non-convertible values become NaN."""
    try:
        return float(str(value).replace(',', ''))
    except ValueError:
        return np.nan  # Return NaN for non-convertible values

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
    The Equity Analysis Calculator helps you evaluate the equity of your URLs by considering various SEO and GA4 metrics.
    It allows you to upload an XLSX file with your URL data and keywords, process the data to calculate weighted scores, classify the URLs,
    and provide actionable recommendations based on their equity scores.
    """)

    st.header("How to Use It")
    st.write("""
    1. **Download the Template**: Click the 'Download Equity Analysis Template XLSX' button to download a template file with the required columns.
    2. **Fill Out the Template**: Enter your values into the template XLSX file. If values aren't applicable, either enter 0 for all empty fields, or delete the header columns.
    3. **Select Columns**: Choose which columns to include in the analysis from the multiselect dropdown. Remove any columns that are empty / unused.
    4. **Upload Your Data**: Use the 'Upload your XLSX file' button to upload your equity analysis data and keywords in separate tabs.
    5. **View Results**: The app will display the analyzed results and provide a download link for the final output file.
    """)

    if st.button("Download Equity Analysis Template XLSX"):
        equity_template_df = pd.DataFrame(columns=[
            "URL", "status_code", "Inlinks", "backlinks", "referring_domains_score", "trust_flow_score",
            "citation_flow_score", "gsc_clicks_score", "gsc_impressions_score",
            "unique_pageviews_organic_score", "unique_pageviews_all_traffic_score", "completed_goals_all_traffic_score",
            "ga4_sessions", "ga4_views", "ga4_engaged_sessions", "ga4_conversions",
            "ga4_views_per_session", "ga4_average_session_duration", "ga4_bounce_rate"
        ])
        keyword_template_df = pd.DataFrame(columns=[
            "URL", "Keywords", "Search Volume", "Ranking Position"
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            equity_template_df.to_excel(writer, sheet_name='Equity Data', index=False)
            keyword_template_df.to_excel(writer, sheet_name='Keyword Data', index=False)
        st.markdown(get_excel_download_link(output, "equity_analysis_template.xlsx"), unsafe_allow_html=True)

    st.header("Expected Outcomes")
    st.write("""
    After processing the data, the tool will:
    - Provide a weighted score for each URL based on the selected metrics.
    - Classify URLs into High, Medium, Low, or No value categories.
    - Offer actionable recommendations for each URL based on their classification.
    """)

    # Update the weights_mapping with new GA4 metrics and adjusted weights
    weights_mapping = {
        # Existing metrics with their weights
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
        "total_search_volume_score": 5,
        # New GA4 metrics with adjusted weights
        "ga4_sessions": 14,
        "ga4_engaged_sessions": 14,
        "ga4_conversions": 16,
        "ga4_views": 6,
        "ga4_views_per_session": 4,
        "ga4_average_session_duration": 4,
        "ga4_bounce_rate": 2
    }

    columns_to_use = st.multiselect(
        "Select columns to use in analysis (unselected columns will be omitted):",
        options=list(weights_mapping.keys()),
        default=list(weights_mapping.keys())
    )

    uploaded_file = st.file_uploader("Upload your XLSX file", type="xlsx")

    if uploaded_file is not None:
        with st.spinner('Processing your file...'):
            try:
                equity_data_df = pd.read_excel(uploaded_file, sheet_name='Equity Data')
                keyword_data_df = pd.read_excel(uploaded_file, sheet_name='Keyword Data')
            except Exception as e:
                st.error(f"Error reading sheets: {e}")
                return

            # Ensure columns are correctly formatted
            equity_data_df.columns = [col.strip().lower().replace(' ', '_') for col in equity_data_df.columns]
            keyword_data_df.columns = [col.strip().lower().replace(' ', '_') for col in keyword_data_df.columns]

            # Fill N/A values with 0
            equity_data_df = equity_data_df.fillna(0)
            keyword_data_df = keyword_data_df.fillna(0)

            # Ensure 'citation_flow_score' has no zero values to avoid division errors
            equity_data_df['citation_flow_score'] = np.where(equity_data_df['citation_flow_score'] == 0, 0.01, equity_data_df['citation_flow_score'])

            # Convert relevant columns to numeric
            analysis_cols_to_numeric = columns_to_use + [
                "total_search_volume_score", "number_of_keywords_page_1_score",
                "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"
            ]
            for col in analysis_cols_to_numeric:
                if col in equity_data_df.columns:
                    equity_data_df[col] = pd.to_numeric(equity_data_df[col].apply(lambda x: str(x).replace(',', '')), errors='coerce')

            # After conversion, fill NaN values with 0
            equity_data_df[analysis_cols_to_numeric] = equity_data_df[analysis_cols_to_numeric].fillna(0)

            # Verify data types
            st.write("Data types of analysis columns:")
            st.write(equity_data_df[analysis_cols_to_numeric].dtypes)

            # Convert keyword data columns to numeric
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

            # Merge keyword data
            if 'url' in equity_data_df.columns and keyword_summary_df is not None and 'url' in keyword_summary_df.columns:
                equity_data_df = equity_data_df.merge(keyword_summary_df, on="url", how="left")
            else:
                st.error("'url' column is missing in one of the uploaded sheets.")
                return

            equity_data_df = equity_data_df.fillna(0)

            # Implement the hybrid weighting approach
            # Calculate averages for each metric
            metric_averages = equity_data_df[columns_to_use].mean()

            weighted_scores_sum = pd.Series(np.zeros(len(equity_data_df)), index=equity_data_df.index)

            for column in columns_to_use:
                if column in equity_data_df.columns:
                    base_weight = weights_mapping[column]
                    average = metric_averages[column]
                    # Avoid division by zero
                    average = average if average != 0 else 0.0001
                    # Adjust weight based on how the URL's metric compares to the average
                    # Assign full weight if value >= average, partial weight if less
                    ratio = equity_data_df[column] / average
                    adjusted_weight = base_weight * np.minimum(ratio, 1)
                    # Optionally, increase weight if significantly above average (e.g., up to 1.5x)
                    adjusted_weight += base_weight * np.maximum((ratio - 1) * 0.5, 0)
                    # Cap the adjusted weight
                    max_weight = base_weight * 1.5  # Adjust as necessary
                    adjusted_weight = adjusted_weight.clip(upper=max_weight)
                    # Normalize the metric
                    min_val = equity_data_df[column].min()
                    max_val = equity_data_df[column].max()
                    if max_val != min_val:
                        normalized_metric = (equity_data_df[column] - min_val) / (max_val - min_val)
                    else:
                        normalized_metric = 0
                    # Add to the weighted sum
                    weighted_scores_sum += normalized_metric * adjusted_weight

            # Include the trust ratio if applicable
            if "trust_flow_score" in columns_to_use and "citation_flow_score" in columns_to_use:
                if (equity_data_df["citation_flow_score"] != 0).all():
                    trust_ratio = (equity_data_df["trust_flow_score"] / equity_data_df["citation_flow_score"]).fillna(0) * 2
                else:
                    trust_ratio = pd.Series(np.zeros(len(equity_data_df)), index=equity_data_df.index)
                weighted_scores_sum += trust_ratio

            # Assign the final weighted score
            equity_data_df["Final_Weighted_Score"] = weighted_scores_sum

            # Proceed with classification as before
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

            # Prepare the final result dataframe
            # Include keyword columns if they were calculated
            keyword_columns = ["total_search_volume_score", "number_of_keywords_page_1_score",
                               "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
            for col in keyword_columns:
                if col in equity_data_df.columns and col not in columns_to_use:
                    columns_to_use.append(col)

            # Exclude 'Final_Weighted_Score' from the export columns if desired
            export_columns = [col for col in equity_data_df.columns if col != "Final_Weighted_Score"]
            result_df = equity_data_df[export_columns]

            result_df = result_df.fillna(0)

        st.write("Classification Distribution:")
        st.write(equity_data_df["Recommendation"].value_counts())

        st.write("Detailed URL Table (showing first 100 rows):")
        st.write(result_df.head(100))

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Results')
        st.markdown(get_excel_download_link(output, "equity_analysis_results.xlsx"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
