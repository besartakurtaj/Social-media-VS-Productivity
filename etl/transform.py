import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": "M", "Female": "F", "Other": "O"})

    #diskretizimi (for numeric columns)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
        df[numeric_cols] = discretizer.fit_transform(df[numeric_cols])
        print("Discretization applied to numeric columns successfully.")

    # Dimensionality reduction
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    selector = VarianceThreshold(threshold=0.01)
    reduced = selector.fit_transform(df[numeric_cols])
    selected_cols = numeric_cols[selector.get_support()]
    df = pd.concat([df[selected_cols], df.drop(columns=numeric_cols, errors="ignore")], axis=1)
