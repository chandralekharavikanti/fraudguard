from pathlib import Path
import pandas as pd

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real ULB data missing: {path}. Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    df = pd.read_csv(path)
    expected = set(FEATURES + ["Class"])
    if set(df.columns) != expected:
        raise ValueError(f"Unexpected columns; missing={sorted(expected-set(df.columns))}, extra={sorted(set(df.columns)-expected)}")
    if df.empty or df[FEATURES + ["Class"]].isna().any().any():
        raise ValueError("Dataset is empty or contains missing values")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("Class must contain both 0 and 1")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be nonnegative")
    return df
