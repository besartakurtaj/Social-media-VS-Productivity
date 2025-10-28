import pandas as pd
from data_quality import assess_data_quality
from data_type_definition import define_data_type

def extract_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = define_data_type(df)
    quality_report = assess_data_quality(df)
    print("Data extracted")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n DataFrame Info:")
    print(df.info())

    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    print(pd.DataFrame({"Missing Count": missing, "Missing %": missing_percent}))

    duplicates = df.duplicated().sum()
    print(f"\n Duplicate Rows: {duplicates}")

    print("\nLogical Data Issues:")
    print(quality_report["logical_issues"])

    print("\n Descriptive Statistics:")
    print(df.describe())

    print("\n Unique Value Counts:")
    for col in df.select_dtypes(include=["object"]).columns[:19]:
        print(f"{col}: {df[col].nunique()} unique values")

    #mostrimi ose sampling (10% of dataset)
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"\nSampled Data Shape: {df_sample.shape}")

    return df