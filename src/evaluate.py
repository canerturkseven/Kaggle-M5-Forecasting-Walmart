#%%
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error


def summarize_cv(df):
    return pd.Series(
        {
            "train_start": df["start_train"].iloc[0],
            "train_end": df["end_train"].iloc[0],
            "forecast_dates": df["date"].unique().tolist(),
            "score": mean_squared_error(df["sales"], df["forecast"]),
        }
    )


runs = mlflow.search_runs(experiment_ids="0")["run_id"].tolist()
for run_id in runs:
    with mlflow.start_run(run_id=run_id) as run:
        try:
            df = pd.read_csv(f"{run.info.artifact_uri}/cross_validation/forecast.csv")
            cv_summary = df.groupby("cv").apply(summarize_cv)
            mlflow.log_metric("rmse", cv_summary["score"].mean())
            mlflow.log_dict(
                cv_summary.to_dict(orient="index"), "cross_valudation/summary_cv.yml"
            )
        except:
            pass


# %%
