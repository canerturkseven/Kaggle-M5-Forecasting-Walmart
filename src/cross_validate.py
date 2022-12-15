#%%
from lightgbm import LGBMRegressor
import pandas as pd
import glob
import yaml
import mlflow
from tqdm.notebook import tqdm
from utils.TimeBasedSplit import TimeBasedSplit
from mlflow.models.signature import infer_signature


def load_training_data():
    df = pd.concat(
        [
            pd.read_parquet(i).assign(date=lambda x: pd.to_datetime(x["date"]))
            for i in glob.glob("../output/feature/*.parquet")
        ]
    )
    df = df[df["date"] < "2016-05-01"].reset_index(drop=True)

    with open("cross_validate_config.yml", "r") as f:
        config = yaml.safe_load(f)

    return df, config


def cross_val_forecast(
    *,
    model,
    cv,
    df,
    date_col,
    feature_cols,
    target_col,
    id_cols,
):
    forecast = []
    for i, fold in enumerate(cv):

        train_idx, test_idx = fold[0], fold[1]
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        model.fit(df_train[feature_cols], df_train[target_col])
        df_test.loc[:, "forecast"] = model.predict(df_test[feature_cols])
        df_test.loc[:, "cv"] = i
        df_test.loc[:, "train_start"] = df_train[date_col].min()
        df_test.loc[:, "train_end"] = df_train[date_col].max()

        forecast.append(
            df_test[
                [
                    *id_cols,
                    date_col,
                    target_col,
                    "forecast",
                    "cv",
                    "train_start",
                    "train_end",
                ]
            ]
        )

    return pd.concat(forecast).reset_index(drop=True)


def cross_validate(df, config):

    id_col = config["data"]["id_col"]
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    date_frequency = config["data"]["date_frequency"]
    n_splits = config["cv"]["splits"]
    test_size = config["cv"]["test_size"]
    hyperparameters = config["hyperparameters"]
    output_file = config["output"]["file"]
    output_dir = config["output"]["dir"]

    with mlflow.start_run() as _:

        model = LGBMRegressor(**hyperparameters)
        mlflow.log_params(hyperparameters)

        cv_forecast = []
        for model_id, model_param in tqdm(config["models"].items()):

            forecast_horizon = model_param["forecast_horizon"]
            feature_cols = model_param["feature_cols"]

            cv = TimeBasedSplit(
                date_col=date_col,
                date_frequency=date_frequency,
                n_splits=n_splits,
                forecast_horizon=forecast_horizon,
                end_offset=test_size - max(forecast_horizon),
                step_length=test_size,
            ).split(df)

            forecast = cross_val_forecast(
                model=model,
                cv=cv,
                df=df,
                date_col=date_col,
                feature_cols=feature_cols,
                target_col=target_col,
                id_cols=[id_col],
            )
            cv_forecast.append(forecast)

            model.fit(df[feature_cols], df[target_col])
            mlflow.lightgbm.log_model(
                model,
                model_id,
                signature=infer_signature(df[feature_cols]),
            )

        pd.concat(cv_forecast).to_csv(f"{output_file}.csv")
        mlflow.log_artifact(f"{output_file}.csv", output_dir)


def main():
    df, config = load_training_data()
    cross_validate(df, config)


if __name__ == "__main__":
    main()

#%%
