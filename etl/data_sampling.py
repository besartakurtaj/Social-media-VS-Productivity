import pandas as pd

def perform_sampling(df: pd.DataFrame, method: str = "random", frac: float = 0.1, random_state: int = 42) -> pd.DataFrame:

    if method == "random":
        # Simple Random Sampling
        df_sample = df.sample(frac=frac, random_state=random_state)
        print(f"Simple Random Sampling ({frac*100:.0f}% of data).")

    elif method == "stratified":
        # Stratified Sampling
        stratify_col = "job_type" if "job_type" in df.columns else None
        if stratify_col:
            df_sample = df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=random_state)
            )
            print(f"Stratified Sampling '{stratify_col}' ({frac*100:.0f}%).")
        else:
            print("No columns available for stratified sampling, random sampling used by default.")
            df_sample = df.sample(frac=frac, random_state=random_state)
    else:
        raise ValueError("Sampling methods should be used: 'random' ose 'stratified'.")

    print(f"Sampling: {df_sample.shape}")
    return df_sample
