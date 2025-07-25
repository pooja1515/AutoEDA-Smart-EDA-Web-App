import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from utils import smart_fill_missing, auto_convert_types, convert_df_to_csv

# 🎨 Setup
st.set_page_config(page_title="AutoEDA+", layout="wide")
st.title("📊 AutoEDA+ - Beautiful & Smart EDA Web App")

# 🌈 Theme toggle
theme = st.radio("Choose Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #0e1117; color: white; }
            .stDataFrame { background-color: #1e1e1e; }
        </style>
    """, unsafe_allow_html=True)

# 📁 Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # 🧼 Clean + convert
        df = auto_convert_types(df)
        df_cleaned = smart_fill_missing(df.copy())

        # 🔍 Preview
        st.subheader("🔍 Cleaned Data Preview")
        st.dataframe(df_cleaned.head())

        # 📥 Download cleaned data
        st.download_button("📥 Download Cleaned CSV", convert_df_to_csv(df_cleaned), file_name="cleaned_data.csv", mime="text/csv")

        # 📋 Dataset Summary
        with st.expander("📊 Dataset Summary"):
            st.write("🧾 Shape:", df_cleaned.shape)
            st.write("🧠 Columns:", df_cleaned.columns.tolist())
            st.write("🔢 Data Types")
            st.dataframe(df_cleaned.dtypes)
            st.write("🕳️ Missing Values (After Filling)")
            st.dataframe(df_cleaned.isnull().sum())
            st.write("📊 Descriptive Statistics")
            st.dataframe(df_cleaned.describe(include='all'))

        # 📉 Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) >= 2:
            fig = plt.figure(figsize=(10, 5))
            sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)

        # 📌 Column Visualizer
        st.subheader("📈 Column Analysis")
        selected_col = st.selectbox("Choose a column", df_cleaned.columns)
        if df_cleaned[selected_col].dtype == 'object' or df_cleaned[selected_col].nunique() < 20:
            fig = px.histogram(df_cleaned, x=selected_col)
        else:
            fig = px.box(df_cleaned, y=selected_col)
        st.plotly_chart(fig, use_container_width=True)

        # # 🧼 Outlier Detection
        # st.subheader("🚨 Outlier Detection")
        # outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
        # if st.checkbox("🧼 Show Outlier Detection (Z-Score > 3)"):
        #     st.subheader("📊 Outlier Count Per Numeric Column")
        #     outliers = detect_outliers(df_cleaned)
        #     st.write(pd.DataFrame(list(outliers.items()), columns=["Feature", "Outlier Count"]))
        #             # Optional: Show boxplots
        # st.subheader("📦 Boxplots for Numeric Columns")
        # for col in df_cleaned.select_dtypes(include=np.number).columns:
        #     fig, ax = plt.subplots()
        #     sns.boxplot(x=df_cleaned[col], ax=ax)
        #     st.pyplot(fig)



        # 🧠 Target vs Feature Analysis
        st.subheader("🎯 Target vs Feature Analysis")
        target_col = st.selectbox("Select target column", df_cleaned.columns)
        feature_col = st.selectbox("Select feature to compare", [col for col in df_cleaned.columns if col != target_col])

        if df_cleaned[target_col].nunique() <= 10:
            fig = px.box(df_cleaned, x=target_col, y=feature_col)
        else:
            fig = px.scatter(df_cleaned, x=feature_col, y=target_col)
        st.plotly_chart(fig, use_container_width=True)

        # 📅 Time Series Detection & Trend Plot
        st.subheader("📈 Time Series Trend")
        datetime_cols = df_cleaned.select_dtypes(include='datetime').columns
        if len(datetime_cols) > 0:
            time_col = st.selectbox("Select datetime column", datetime_cols)
            metric_col = st.selectbox("Select numeric column to analyze over time", numeric_cols)
            ts_df = df_cleaned[[time_col, metric_col]].dropna().sort_values(time_col)
            fig = px.line(ts_df, x=time_col, y=metric_col, title=f"{metric_col} over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datetime column detected.")

        # ⚙️ Custom Chart Builder
        st.subheader("⚙️ Custom Chart Builder")
        x_col = st.selectbox("X-Axis", df_cleaned.columns, key="x")
        y_col = st.selectbox("Y-Axis", df_cleaned.columns, key="y")
        chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter"])
        if chart_type == "Line":
            fig = px.line(df_cleaned, x=x_col, y=y_col)
        elif chart_type == "Bar":
            fig = px.bar(df_cleaned, x=x_col, y=y_col)
        else:
            fig = px.scatter(df_cleaned, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
