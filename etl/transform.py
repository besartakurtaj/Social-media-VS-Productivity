import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
