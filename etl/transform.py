import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from dependency_map import dependency_map
from missingValues import advanced_imputation
from binarization import apply_binarization
from aggregation import add_aggregated
from features import create_features
from discretization import apply_discretization

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    #Missing Value Imputation
    df = advanced_imputation(df, dependency_map)
    
    # Binarization
    df = apply_binarization(df)

    # Aggregation
    df = add_aggregated(df)
    df = create_features(df)

    #Discretization
    df = apply_discretization(df, column="daily_social_media_time", n_bins=4, strategy="uniform")

    return df 

    # # Dimensionality reduction
    # numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    # selector = VarianceThreshold(threshold=0.01)
    # reduced = selector.fit_transform(df[numeric_cols])
    # selected_cols = numeric_cols[selector.get_support()]
    # df = pd.concat([df[selected_cols], df.drop(columns=numeric_cols, errors="ignore")], axis=1)
