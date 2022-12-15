#%%
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error


def summarize_cv(df):
    return pd.Series(
        {
            "train_start": df["train_start"].iloc[0],
            "train_end": df["train_end"].iloc[0],
            "forecast_start": df["date"].min(),
            "forecast_end": df["date"].max(),
            "rmse": mean_squared_error(df["sales"], df["forecast"], squared=False),
        }
    )


def evaluate():
    runs = mlflow.search_runs()["run_id"].tolist()
    for run_id in runs:

        with mlflow.start_run(run_id=run_id) as run:
            if not run.data.metrics:
                df = pd.read_csv(f"{run.info.artifact_uri}/cv/forecast.csv")
                df_metric = df.groupby("cv", as_index=False).apply(summarize_cv)
                df_metric.to_csv("metric.csv")

                mlflow.log_metric(
                    "rmse",
                    df_metric["rmse"].mean(),
                )
                mlflow.log_artifact(
                    "metric.csv",
                    artifact_path="cv",
                )


def main():
    evaluate()


if __name__ == "__main__":
    main()

# %%
