import pandas as pd

def extract_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)