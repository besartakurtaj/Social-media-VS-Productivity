import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": "M", "Female": "F", "Other": "O"})