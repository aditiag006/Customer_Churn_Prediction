"""
preprocessing.py
Reusable function to transform raw Telco customer data into the exact processed format the the trained model expects.
It mirrors every step from 02_preprocessing_features.ipynb, in the same order
"""

import pandas as pd
import numpy as np

# Columns where "No internet service" / "No phone service" get collapsed to "No"
COLS_TO_FIX = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

BINARY_YESNO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + COLS_TO_FIX

FINAL_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'InternetService_Fiber optic', 'InternetService_No',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'TotalServices',
    'TenureGroup_1-2yr', 'TenureGroup_2-4yr', 'TenureGroup_4-6yr'
]


def preprocess_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw dataframe matching the original Kaggle Telco CSV's columns
    (one or many rows) and returns it transformed into the model's expected
    27-column processed format (before scaling).
    """
    df = df_raw.copy()

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target if present (only relevant for batch CSVs that include it)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Consolidate redundant categories
    for col in COLS_TO_FIX:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # Binary mapping
    binary_map_yesno = {"Yes": 1, "No": 0}
    gender_map = {"Female": 0, "Male": 1}

    for col in BINARY_YESNO_COLS:
        df[col] = df[col].map(binary_map_yesno)
    df["gender"] = df["gender"].map(gender_map)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    # Feature engineering
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["TotalServices"] = df[service_cols].sum(axis=1)

    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    df = pd.get_dummies(df, columns=["TenureGroup"], drop_first=True)

    # Ensure every expected column exists (e.g. a single customer's CSV
    # row might not trigger every possible one-hot category), filling
    # missing ones with 0, and drop anything unexpected.
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FINAL_COLUMNS]  # enforce exact column order

    return df
