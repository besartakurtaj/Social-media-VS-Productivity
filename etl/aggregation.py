import pandas as pd

def add_aggregated(df: pd.DataFrame) -> pd.DataFrame:

    job_avg = df.groupby("job_type")["perceived_productivity_score"].mean().reset_index()
    job_avg.rename(columns={"perceived_productivity_score": "avg_perceived_prod"}, inplace=True)

    global_avg = df["perceived_productivity_score"].mean()

    job_avg["job_optimism"] = pd.cut(
        job_avg["avg_perceived_prod"],
        bins=[-999, global_avg - 0.5, global_avg + 0.5, 999],
        labels=["Pessimistic Job", "Neutral Job", "Optimistic Job"]
    )

    df = df.merge(job_avg[["job_type", "job_optimism"]], on="job_type", how="left")

    return df
