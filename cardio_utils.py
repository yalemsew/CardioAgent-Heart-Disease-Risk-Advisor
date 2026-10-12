"""Shared data preparation for the cardio project.

These are the same steps explained one-by-one in 01_cardio_basics.ipynb (Steps 1-7),
collected in one place so every notebook prepares the data in exactly the same way.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "cardio_train.csv"

# Features the patient can realistically change (used by the agent's what-if analysis)
MODIFIABLE = ["ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]


def load_clean_data(path=DATA_PATH):
    """Steps 1-4: load, remove impossible values, add age_years / bmi / is_male."""
    df = pd.read_csv(path, sep=";")
    keep = (
        df["ap_hi"].between(80, 250)
        & df["ap_lo"].between(40, 160)
        & (df["ap_hi"] > df["ap_lo"])
        & df["height"].between(130, 210)
        & df["weight"].between(35, 200)
    )
    df = df[keep].copy()
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df["is_male"] = (df["gender"] == 2).astype(int)
    return df.drop(columns=["id", "age", "height", "weight", "gender"])


def split_and_scale(df):
    """Steps 6-7: 70/15/15 stratified split, scaler fitted on TRAIN only.

    Returns a dict with raw splits (X_train, ...), scaled splits (X_train_s, ...),
    targets (y_train, ...), the fitted scaler, and the feature names.
    """
    X = df.drop(columns=["cardio"])
    y = df["cardio"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_s": scaler.transform(X_train),
        "X_val_s": scaler.transform(X_val),
        "X_test_s": scaler.transform(X_test),
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "features": list(X.columns),
    }
