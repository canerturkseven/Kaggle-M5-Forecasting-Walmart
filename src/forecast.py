#%%
from lightgbm import LGBMRegressor
from utils.TimeBasedSplit import TimeBasedSplit
import pandas as pd
import glob
import yaml
from sklearn.metrics import mean_squared_error
import mlflow
from tqdm.notebook import tqdm


def load_training_data():
    df = pd.concat(
        [
            pd.read_parquet(i).assign(date=lambda x: pd.to_datetime(x["date"]))
            for i in glob.glob("../output/feature/*.parquet")
        ]
    )
    df = df[df["date"] < "2016-05-01"].reset_index(drop=True)

    with open("forecast_config.yml", "r") as f:
        config = yaml.safe_load(f)

    return df, config


def cross_validation(df, config):

    target_col = config["data"]["target_col"]
    date_col = config["data"]["target_col"]
    date_frequency = config["data"]["target_col"]
    id_col = config["data"]["id_col"]
    output_file = config["data"]["output"]["name"]
    output_dir = config["data"]["output"]["dir"]

    for model_id, model_config in tqdm(config["models"].items()):
        with mlflow.start_run(run_name=model_id) as _:

            features = model_config["features"]
            forecast_horizon = model_config["forecast_horizon"]
            hyperparameters = model_config["hyperparameters"]
            cv_splits = model_config["cv_splits"]

            cv_folds = TimeBasedSplit(
                date_col=date_col,
                date_frequency=date_frequency,
                n_splits=cv_splits,
                test_size=len(forecast_horizon),
                gap=min(forecast_horizon) - 1,
            ).split(df)

            df_forecast = []
            for i, fold in tqdm(enumerate(cv_folds)):

                train_idx, test_idx = fold[0], fold[1]
                df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

                mlflow.lightgbm.autolog()
                model = LGBMRegressor(**hyperparameters)
                model.fit(df_train[features], df_train[target_col])

                df_test["forecast"] = model.predict(df_test[features])
                df_test["train_start"] = df_train[date_col].min()
                df_test["train_end"] = df_train[date_col].max()
                df_test["cv"] = i
                df_forecast.append(
                    df_test[
                        [
                            id_col,
                            date_col,
                            target_col,
                            "cv",
                            "forecast",
                            "train_start",
                            "train_end",
                        ]
                    ]
                )

            model.fit(df[features], df[target_col])
            pd.concat(df_forecast).to_csv(f"{output_file}.csv")
            mlflow.log_artifact(f"{output_file}.csv", output_dir)


def main():
    df, config = load_training_data()
    cross_validation(df, config)


if __name__ == "__main__":
    main()
#%%
