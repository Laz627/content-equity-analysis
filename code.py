import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

# Check if openpyxl is installed
try:
    import openpyxl
except ImportError:
    st.error("The openpyxl library is required to handle Excel files. Please install it using: pip install openpyxl")
    st.stop()

def convert_to_numeric(value):
    """Converts a value to numeric format after replacing commas."""
    try:
        return float(str(value).replace(',', ''))
    except ValueError:
        return np.nan  # Return NaN for non-convertible values

def classify_score(row, high_thresh, med_thresh, metric_cols):
    """Classify the score into High, Medium, Low, or No value based on thresholds and metric values."""
    # Check if all selected metrics are zero
    if (row[metric_cols] == 0).all():
        return "No value"
    elif row["Final_Weighted_Score"] >= high_thresh:
        return "High"
    elif row["Final_Weighted_Score"] >= med_thresh:
        return "Medium"
    else:
        return "Low"

def get_excel_download_link(output, filename):
    """Generates a link to download the DataFrame as an Excel file."""
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href

def z_score_normalize(df, metric_cols):
    """Normalize the specified columns in the DataFrame using z-score normalization."""
    df_normalized = df.copy()
    for col in metric_cols:
        if col in df_normalized.columns:
            mean = df_normalized[col].mean()
            std = df_normalized[col].std()
            if std != 0:
                df_normalized[col] = (df_normalized[col] - mean) / std
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
            "GA4_Sessions", "GA4_Views", "GA4_Engaged_Sessions", "GA4_Conversions",
            "GA4_Views_Per_Session", "GA4_Average_Session_Duration", "GA4_Bounce_Rate"
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

    # Weights mapping with GA4 metrics
    weights_mapping = {
        "inlinks": 4,
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
        "ga4_sessions": 14,               # Important
        "ga4_engaged_sessions": 12,       # Important
        "ga4_conversions": 12,            # Important
        "ga4_views_per_session": 5,       # Less important
        "ga4_average_session_duration": 5,  # Less important
        "ga4_bounce_rate": 5,             # Less important
        "ga4_views": 4                    # Less important
    }

    # Default selected columns, including new GA4 metrics
    default_columns = list(weights_mapping.keys())

    columns_to_use = st.multiselect(
        "Select columns to use in analysis (unselected columns will be omitted):",
        options=default_columns,
        default=default_columns
    )

    # Convert columns_to_use to lowercase
    columns_to_use = [col.lower() for col in columns_to_use]

    uploaded_file = st.file_uploader("Upload your XLSX file", type="xlsx")

    if uploaded_file is not None:
        with st.spinner('Processing your file...'):
            try:
                equity_data_df = pd.read_excel(uploaded_file, sheet_name='Equity Data')
                keyword_data_df = pd.read_excel(uploaded_file, sheet_name='Keyword Data')
            except Exception as e:
                st.error(f"Error reading sheets: {e}")
                return

            # Standardize column names
            equity_data_df.columns = [col.strip().lower().replace(' ', '_') for col in equity_data_df.columns]
            keyword_data_df.columns = [col.strip().lower().replace(' ', '_') for col in keyword_data_df.columns]

            # Fill N/A values with 0 in both dataframes
            equity_data_df = equity_data_df.fillna(0)
            keyword_data_df = keyword_data_df.fillna(0)

            # Ensure 'citation_flow_score' has no zero values to avoid division errors
            equity_data_df['citation_flow_score'] = np.where(equity_data_df['citation_flow_score'] == 0, 0.01, equity_data_df['citation_flow_score'])

            # Identify time duration columns
            time_duration_columns = ['ga4_average_session_duration']  # Add any other time duration columns here

            # Convert time duration strings to total seconds
            for col in time_duration_columns:
                if col in equity_data_df.columns:
                    equity_data_df[col] = pd.to_timedelta(equity_data_df[col].astype(str), errors='coerce').dt.total_seconds()
                    equity_data_df[col] = equity_data_df[col].fillna(0)

            # Convert analysis columns to numeric
            analysis_cols_to_numeric = columns_to_use + [
                "total_search_volume_score",
                "number_of_keywords_page_1_score",
                "number_of_keywords_page_2_score",
                "number_of_keywords_page_3_score"
            ]

            for col in analysis_cols_to_numeric:
                if col in equity_data_df.columns:
                    equity_data_df[col] = equity_data_df[col].apply(convert_to_numeric)
                    equity_data_df[col] = equity_data_df[col].fillna(0)

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

            if 'url' in equity_data_df.columns and 'url' in keyword_summary_df.columns:
                equity_data_df = equity_data_df.merge(keyword_summary_df, on="url", how="left")
            else:
                st.error("'url' column is missing in one of the uploaded sheets.")
                return

            equity_data_df = equity_data_df.fillna(0)

            # Z-score normalization of selected columns
            norm_data_df = z_score_normalize(equity_data_df.copy(), columns_to_use)

            # Calculate weighted scores
            weighted_scores_sum = pd.Series(np.zeros(len(norm_data_df)), index=norm_data_df.index)
            for column in columns_to_use:
                if column in norm_data_df.columns:
                    weight = weights_mapping.get(column, 0)
                    weighted_sum = norm_data_df[column] * weight
                    weighted_scores_sum += weighted_sum
                    # Debugging output
                    equity_data_df[f"{column}_weighted"] = norm_data_df[column] * weight

            # Calculate trust ratio if applicable
            if "trust_flow_score" in columns_to_use and "citation_flow_score" in columns_to_use:
                if (norm_data_df["citation_flow_score"] != 0).all():
                    trust_ratio = (norm_data_df["trust_flow_score"] / norm_data_df["citation_flow_score"]).fillna(0) * 2
                else:
                    trust_ratio = pd.Series(np.zeros(len(norm_data_df)), index=norm_data_df.index)
                weighted_scores_sum += trust_ratio
                # Debugging output
                equity_data_df["trust_ratio"] = trust_ratio

            equity_data_df["Final_Weighted_Score"] = weighted_scores_sum

            # Determine thresholds for classification
            high_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.85)
            medium_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.50)

            # Classification
            equity_data_df["Recommendation"] = equity_data_df.apply(
                classify_score, axis=1, args=(high_threshold, medium_threshold, columns_to_use)
            )

            action_mapping = {
                "High": "Keep or Maintain 1:1 redirect",
                "Medium": "Update or consolidate content",
                "Low": "Evaluate URL priority or deprecate",
                "No value": "Do not keep or migrate"
            }
            equity_data_df["Action"] = equity_data_df["Recommendation"].map(action_mapping)

            # Include keyword columns if not already selected
            keyword_columns = ["total_search_volume_score", "number_of_keywords_page_1_score",
                               "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
            for col in keyword_columns:
                if col in equity_data_df.columns and col not in columns_to_use:
                    columns_to_use.append(col)

            # Prepare final result DataFrame
            export_columns = [col for col in equity_data_df.columns if not col.endswith("_weighted") and col not in ("Final_Weighted_Score", "trust_ratio")]
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

    else:
        st.write("Please upload an XLSX file to proceed.")

if __name__ == "__main__":
    main()
