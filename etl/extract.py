import pandas as pd

def extract_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("Data extracted")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n DataFrame Info:")
    print(df.info())

    print("\n Missing Values per Column:")
    print(df.isnull().sum())

    duplicates = df.duplicated().sum()
    print(f"\n Duplicate Rows: {duplicates}")

    print("\n Descriptive Statistics:")
    print(df.describe())

    print("\n Unique Value Counts:")
    for col in df.select_dtypes(include=["object"]).columns[:19]:
        print(f"{col}: {df[col].nunique()} unique values")

    #mostrimi ose sampling (10% of dataset)
    df_sample = df.sample(frac=0.1, random_state=42)
    print(f"\nSampled Data Shape: {df_sample.shape}")

    return df