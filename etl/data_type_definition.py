import pandas as pd

TYPE_MAPPING = {
    "age": "Int64",
    "number_of_notifications": "Int64",
    "coffee_consumption_per_day": "Int64",
    "gender": "category",
    "job_type": "category",
    "social_platform_preference": "category",
    "uses_focus_apps": "Int64",
    "has_digital_wellbeing_enabled": "Int64",
    "daily_social_media_time": "float64",
    "work_hours_per_day": "float64",
    "perceived_productivity_score": "float64",
    "actual_productivity_score": "float64",
    "stress_level": "Int64", 
    "sleep_hours": "float64",
    "screen_time_before_sleep": "float64",
    "breaks_during_work": "Int64",
    "days_feeling_burnout_per_month": "Int64",
    "weekly_offline_hours": "float64",
    "job_satisfaction_score": "float64"
}

def define_data_type(df: pd.DataFrame, type_map: dict = TYPE_MAPPING) -> pd.DataFrame:

    df_new = df.copy()
    
    for col, data_type in type_map.items():
        if col in df_new.columns:
            try:
                if data_type in ["Int64", "category"]:
                     df_new[col] = df_new[col].astype(data_type)
                else:
                    df_new[col] = df_new[col].convert_dtypes(convert_string=False).astype(data_type)
            except Exception as e:
                print(f"Warning: Cannot convert '{col}' to '{data_type}'. Error: {e}")

    print("\nData Type Definitions:")
    print(df_new.dtypes.to_string())

    return df_new