import streamlit as st
import pandas as pd
import numpy as np

@st.cache
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

@st.cache
def get_table_download_link(df, filename):
    """Generates a link to download the DataFrame as a CSV file."""
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

def main():
    st.title("Equity Analysis Calculator")
    st.write("**Created by: Brandon Lazovic**")

    st.header("How It Works")
    st.write("""
    The Equity Analysis Calculator helps you evaluate the equity of your URLs by considering various SEO metrics.
    It allows you to upload a CSV file with your URL data, process the data to calculate weighted scores, classify the URLs,
    and provide actionable recommendations based on their equity scores. Additionally, it supports the evaluation of keyword rankings
    if a keyword CSV file is provided.
    """)

    st.header("How to Use It")
    st.write("""
    1. **Download the Template**: Click the 'Download Equity Analysis Template CSV' or 'Download Keyword Template CSV' buttons to download template files with the required columns.
    2. **Upload Your Data**: Use the 'Upload your CSV file' button to upload your equity analysis data. Optionally, upload a keyword data file using 'Upload your keyword CSV file' for additional keyword analysis.
    3. **Select Columns**: Choose which columns to include in the analysis from the multiselect dropdown.
    4. **View Results**: The app will display the analyzed results and provide a download link for the final output file.
    """)

    st.header("Expected Outcomes")
    st.write("""
    After processing the data, the tool will:
    - Provide a weighted score for each URL based on the selected metrics.
    - Classify URLs into High, Medium, Low, or No value categories.
    - Offer actionable recommendations for each URL based on their classification.
    - Optionally, analyze and display keyword ranking metrics if a keyword file is provided.
    """)

    # Allow users to download a template CSV file for equity analysis
    if st.button("Download Equity Analysis Template CSV"):
        equity_template_df = pd.DataFrame(columns=[
            "URL", "status_code", "Inlinks", "backlinks", "referring_domains_score", "trust_flow_score", 
            "citation_flow_score", "gsc_clicks_score", "gsc_impressions_score", 
            "unique_pageviews_organic_score", "unique_pageviews_all_traffic_score", "completed_goals_all_traffic_score"
        ])
        st.markdown(get_table_download_link(equity_template_df, "equity_analysis_template.csv"), unsafe_allow_html=True)

    # Allow users to download a template CSV file for keyword analysis
    if st.button("Download Keyword Template CSV"):
        keyword_template_df = pd.DataFrame(columns=[
            "URL", "Keywords", "Search Volume", "Ranking Position"
        ])
        st.markdown(get_table_download_link(keyword_template_df, "keyword_template.csv"), unsafe_allow_html=True)

    # Allow users to upload their own CSV file
    uploaded_file = st.file_uploader("Upload your Equity Analysis CSV file", type="csv")
    uploaded_keyword_file = st.file_uploader("Upload your Keyword CSV file", type="csv")

    if uploaded_file is not None:
        equity_data_df = pd.read_csv(uploaded_file)

        # Adjusted weights mapping based on the dataset's columns -- add up to 100 points
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
            "completed_goals_all_traffic_score": 6
        }

        # Option to omit missing columns from weighting
        columns_to_use = st.multiselect(
            "Select columns to use in analysis (unselected columns will be omitted):",
            options=list(weights_mapping.keys()),
            default=list(weights_mapping.keys())
        )

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

        # Calculating the weighted scores for each metric in the adjusted dataset
        for column in columns_to_use:
            if column in equity_data_df.columns:
                weight = weights_mapping[column]
                equity_data_df[f"{column}_Weighted"] = equity_data_df[column] * weight

        # Calculate Trust Ratio Score as trust_flow_score / citation_flow_score
        if "trust_flow_score" in columns_to_use and "citation_flow_score" in columns_to_use:
            equity_data_df["Trust_Ratio_Weighted"] = (equity_data_df["trust_flow_score"] / equity_data_df["citation_flow_score"]).fillna(0) * 2

        # Columns to include for the final weighted score
        columns_for_final_score = [f"{col}_Weighted" for col in columns_to_use if col in equity_data_df.columns] + ["Trust_Ratio_Weighted"]

        # Compute the final weighted score for each URL
        equity_data_df["Final_Weighted_Score"] = equity_data_df[columns_for_final_score].sum(axis=1)

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

        # Merge with keyword data if provided
        if uploaded_keyword_file is not None:
            keyword_data_df = pd.read_csv(uploaded_keyword_file)
            required_keyword_columns = ["URL", "Keywords", "Search Volume", "Ranking Position"]
            if all(col in keyword_data_df.columns for col in required_keyword_columns):
                keyword_summary_df = keyword_data_df.groupby("URL").agg({
                    "Search Volume": "sum",
                    "Ranking Position": lambda x: {
                        "Page 1": (x <= 10).sum(),
                        "Page 2": ((x > 10) & (x <= 20)).sum(),
                        "Page 3": ((x > 20) & (x <= 30)).sum()
                    }
                }).reset_index()

                # Normalize the keyword summary data to correct column structure
                keyword_summary_df = pd.DataFrame(keyword_summary_df.to_dict()['Ranking Position'].tolist(), index=keyword_summary_df['URL']).reset_index()
                keyword_summary_df.columns = ["URL", "number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]
                keyword_summary_df["total_search_volume_score"] = keyword_data_df.groupby("URL")["Search Volume"].sum().values

                equity_data_df = equity_data_df.merge(keyword_summary_df, on="URL", how="left")

                # Correct data formatting for keyword columns
                for col in ["total_search_volume_score", "number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score"]:
                    equity_data_df[col] = equity_data_df[col].apply(convert_to_numeric)

                # Recalculate the final weighted score with keyword data
                for column in ["number_of_keywords_page_1_score", "number_of_keywords_page_2_score", "number_of_keywords_page_3_score", "total_search_volume_score"]:
                    if column in columns_to_use and column in equity_data_df.columns:
                        weight = weights_mapping.get(column.split("_score")[0], 0)
                        equity_data_df[f"{column}_Weighted"] = equity_data_df
                        equity_data_df[f"{column}_Weighted"] = equity_data_df[column] * weight

                columns_for_final_score = [f"{col}_Weighted" for col in columns_to_use if col in equity_data_df.columns] + ["Trust_Ratio_Weighted"]
                equity_data_df["Final_Weighted_Score"] = equity_data_df[columns_for_final_score].sum(axis=1)

                # Reclassify based on the new scores
                high_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.85)
                medium_threshold = equity_data_df["Final_Weighted_Score"].quantile(0.50)

                equity_data_df["Recommendation"] = equity_data_df["Final_Weighted_Score"].apply(lambda x: classify_score(x, high_threshold, medium_threshold))
                equity_data_df["Action"] = equity_data_df["Recommendation"].map(action_mapping)

        # Remove weighted score columns from the export
        export_columns = [col for col in equity_data_df.columns if not col.endswith('_Weighted')]
        result_df = equity_data_df[export_columns]

        # Show the results
        st.write(result_df)

        # Allow users to download the resulting DataFrame
        st.markdown(get_table_download_link(result_df, "equity_analysis_results.csv"), unsafe_allow_html=True)

    if uploaded_keyword_file is not None and uploaded_file is None:
        st.error("Please upload the Equity Analysis CSV file first.")

if __name__ == "__main__":
    main()
