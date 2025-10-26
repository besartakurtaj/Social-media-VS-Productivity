import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def apply_binarization(df: pd.DataFrame) -> pd.DataFrame:

    binary_cols = ["uses_focus_apps", "has_digital_wellbeing_enabled"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)  # True -> 1, False -> 0

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1, "Other": 2})

    nominal_cols = ["social_platform_preference"]
    nominal_cols = [col for col in nominal_cols if col in df.columns]

    if len(nominal_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(df[nominal_cols])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(nominal_cols)
        )
        df = pd.concat([df.drop(columns=nominal_cols), encoded_df], axis=1)

    if "stress_level" in df.columns:
        df["high_stress"] = (df["stress_level"] >= 6).astype(int)

    if "daily_social_media_time" in df.columns:
        df["social_addicted"] = (df["daily_social_media_time"] > 4).astype(int)

    if "sleep_hours" in df.columns:
        df["low_sleep"] = (df["sleep_hours"] < 6).astype(int)

    if "number_of_notifications" in df.columns:
        df["too_many_notifications"] = (df["number_of_notifications"] > 50).astype(int)

    if "days_feeling_burnout_per_month" in df.columns:
        df["burnout_risk"] = (df["days_feeling_burnout_per_month"] >= 10).astype(int)

    return df
