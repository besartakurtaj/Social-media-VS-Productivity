import pandas as pd

def advanced_imputation(df: pd.DataFrame, dependency_map: dict) -> pd.DataFrame:

    impute_report = {}

    for target_col, ref_cols in dependency_map.items():
        if target_col not in df.columns:
            continue

        missing_indices = df[df[target_col].isna()].index
        
        if len(missing_indices) == 0:
            continue

        for idx in missing_indices:
            row = df.loc[idx]

            group_filter = pd.Series([True] * len(df))
            for ref_col in ref_cols:
                if ref_col in df.columns and pd.notna(row[ref_col]):
                    if df[ref_col].dtype == "float64":
                        group_filter &= df[ref_col].between(row[ref_col] - 1, row[ref_col] + 1)
                    else:
                        group_filter &= (df[ref_col] == row[ref_col])

            group_values = df.loc[group_filter, target_col].dropna()

            if len(group_values) > 0:
                df.at[idx, target_col] = group_values.median()
            else:
                df.at[idx, target_col] = df[target_col].median()

        impute_report[target_col] = len(missing_indices)
    print(impute_report)

    return df