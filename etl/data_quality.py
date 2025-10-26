import pandas as pd

def assess_data_quality(df: pd.DataFrame) -> dict:
    report = {}

    # Logical validity checks
    logical_issues = {}

    # Age realistic
    if "age" in df.columns:
        logical_issues["unrealistic_age"] = df[(df["age"] < 15) | (df["age"] > 90)].shape[0]

    # Social media time realistic (max 24h)
    if "daily_social_media_time" in df.columns:
        logical_issues["invalid_social_media_time"] = df[df["daily_social_media_time"] > 24].shape[0]

    # Sleep rules
    if "screen_time_before_sleep" in df.columns and "sleep_hours" in df.columns:
        logical_issues["screen_time_exceeds_sleep"] = df[
            df["screen_time_before_sleep"] > df["sleep_hours"]
        ].shape[0]

    # Work time realistic
    if "work_hours_per_day" in df.columns:
        logical_issues["invalid_work_hours"] = df[
            (df["work_hours_per_day"] < 0) | (df["work_hours_per_day"] > 24)
        ].shape[0]

    # Stress range
    if "stress_level" in df.columns:
        logical_issues["invalid_stress_level_range"] = df[
            (df["stress_level"] < 0) | (df["stress_level"] > 10)
        ].shape[0]

    # Job satisfaction range
    if "job_satisfaction_score" in df.columns:
        logical_issues["invalid_job_satisfaction"] = df[
            (df["job_satisfaction_score"] < 0) | (df["job_satisfaction_score"] > 10)
        ].shape[0]

    # Coffee per day
    if "coffee_consumption_per_day" in df.columns:
        logical_issues["extreme_coffee_consumption"] = df[
            df["coffee_consumption_per_day"] > 15
        ].shape[0]

    # Burnout realistic
    if "days_feeling_burnout_per_month" in df.columns:
        logical_issues["invalid_burnout_days"] = df[
            df["days_feeling_burnout_per_month"] > 31
        ].shape[0]

    report["logical_issues"] = pd.DataFrame.from_dict(
        logical_issues, orient="index", columns=["Invalid Count"]
    )

    return report
