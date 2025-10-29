import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (
        a.astype(float) / b.replace({0: np.nan})
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # stress_sleep_ratio, insomnia_pressure
    if {"stress_level", "sleep_hours"}.issubset(df.columns):
        df["stress_sleep_ratio"] = _safe_div(df["stress_level"], df["sleep_hours"])
        # përdor deficitin e gjumit pa krijuar kolonë të veçantë
        sleep_deficit = (8 - df["sleep_hours"]).clip(lower=0)
        df["insomnia_pressure"] = df["stress_level"] * sleep_deficit

    # mins_per_notification, distraction_load
    if {"daily_social_media_time", "number_of_notifications"}.issubset(df.columns):
        df["mins_per_notification"] = _safe_div(
            df["daily_social_media_time"] * 60,
            df["number_of_notifications"]
        )
        df["distraction_load"] = df["daily_social_media_time"] * df["number_of_notifications"]

    # work_to_social_ratio, overbooked_hours
    if {"work_hours_per_day", "daily_social_media_time"}.issubset(df.columns):
        df["work_to_social_ratio"] = _safe_div(
            df["work_hours_per_day"], df["daily_social_media_time"]
        )
        if "sleep_hours" in df.columns:
            df["overbooked_hours"] = (
                df["work_hours_per_day"] + df["daily_social_media_time"] + df["sleep_hours"] - 24
            )

    # burnout_rate
    if "days_feeling_burnout_per_month" in df.columns:
        df["burnout_rate"] = _safe_div(
            df["days_feeling_burnout_per_month"], pd.Series(30, index=df.index)
        )

    return df

# def select_features(df: pd.DataFrame,
#                     variance_threshold: float = 0.001,
#                     corr_threshold: float = 0.95) -> pd.DataFrame:
    
#     numeric = df.select_dtypes(include=[np.number])
#     if not numeric.empty:
#         vt = VarianceThreshold(threshold=variance_threshold)
#         vt.fit(numeric)
#         keep_cols = numeric.columns[vt.get_support(indices=True)]
#         df = df[keep_cols]

#     corr = df.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
#     df = df.drop(columns=to_drop, errors="ignore")

#     return df
