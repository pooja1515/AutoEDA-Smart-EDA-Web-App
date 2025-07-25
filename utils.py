import pandas as pd

import pandas as pd
import numpy as np

def auto_convert_types(df):
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric conversion first
            try:
                df[col] = pd.to_numeric(df[col])
                continue
            except:
                pass

            # Try datetime only if 80%+ values can be parsed
            try:
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                parse_ratio = parsed_dates.notna().mean()
                if parse_ratio > 0.8:
                    df[col] = parsed_dates
            except:
                pass
    return df


def smart_fill_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# def detect_outliers(df, z_thresh=3):
#     numeric_cols = df.select_dtypes(include=np.number).columns
#     outlier_summary = {}
#     for col in numeric_cols:
#         try:
#             z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
#             outlier_summary[col] = (z_scores > z_thresh).sum()
#         except Exception:
#             outlier_summary[col] = 'Error'
#     return outlier_summary
