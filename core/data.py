"""Data loading + preprocessing.

`preprocess()` mirrors the notebook step by step (same encodings, same column
order) so the app's model behaves exactly like the one trained in the notebook.
Decision trees are invariant to feature scaling, so StandardScaler is skipped
here - this keeps tree thresholds in real units (e.g. tenure <= 17.5 months).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "Telco-Customer-Churn.csv"

CATS = {
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
YES_NO = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

ALLOWED = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    **{c: {"Yes", "No"} for c in YES_NO},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": set(CATS["InternetService"]),
    **{c: {"Yes", "No", "No internet service"} for c in ADDONS},
    "Contract": set(CATS["Contract"]),
    "PaymentMethod": set(CATS["PaymentMethod"]),
}


def pretty(name: str) -> str:
    return (name.replace("InternetService_", "Internet: ")
                .replace("Contract_", "Contract: ")
                .replace("PaymentMethod_", "Payment: "))


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def total_charges_median() -> float:
    return float(pd.to_numeric(load_raw()["TotalCharges"], errors="coerce").median())


def preprocess(raw: pd.DataFrame, total_median: float) -> pd.DataFrame:
    """Raw Telco dataframe -> fully numeric dataframe (same steps as the notebook)."""
    d = raw.copy()
    if "customerID" in d.columns:
        d = d.drop(columns=["customerID"])

    d["gender"] = d["gender"].map({"Female": 0, "Male": 1})
    for c in YES_NO:
        d[c] = d[c].map({"No": 0, "Yes": 1})
    d["MultipleLines"] = (d["MultipleLines"].replace("No phone service", "No")
                          .map({"No": 0, "Yes": 1}))

    for col in ["InternetService"]:
        d[col] = pd.Categorical(d[col], categories=CATS[col])
    d = pd.get_dummies(d, columns=["InternetService"], dtype=int)

    for c in ADDONS:
        d[c] = d[c].replace("No internet service", "No").map({"No": 0, "Yes": 1})

    for col in ["Contract", "PaymentMethod"]:
        d[col] = pd.Categorical(d[col], categories=CATS[col])
        d = pd.get_dummies(d, columns=[col], dtype=int)

    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(total_median)
    if "Churn" in d.columns:
        d["Churn"] = d["Churn"].map({"No": 0, "Yes": 1})
    return d


def validate_raw(df: pd.DataFrame):
    """Returns (missing_columns, {column: [unexpected values]})."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    bad = {}
    for col, allowed in ALLOWED.items():
        if col in df.columns:
            vals = set(df[col].dropna().unique().tolist())
            if col == "SeniorCitizen":
                vals = {int(v) if str(v).isdigit() else v for v in vals}
            extra = sorted(map(str, vals - allowed))
            if extra:
                bad[col] = extra
    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").isna().any():
            bad[col] = ["non-numeric / empty values"]
    return missing, bad


@st.cache_data(show_spinner=False)
def get_encoded() -> pd.DataFrame:
    return preprocess(load_raw(), total_charges_median())


@st.cache_data(show_spinner=False)
def get_split(test_size: float = 0.2, seed: int = 42):
    enc = get_encoded()
    X = enc.drop(columns=["Churn"])
    y = enc["Churn"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


@st.cache_data(show_spinner=False)
def get_eda() -> pd.DataFrame:
    """Raw data with friendly extra columns for charts."""
    d = load_raw().copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(total_charges_median())
    d["ChurnFlag"] = (d["Churn"] == "Yes").astype(int)
    d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})
    d["TenureGroup"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                              labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]).astype(str)
    return d


def feature_names() -> list:
    return list(get_encoded().drop(columns=["Churn"]).columns)
