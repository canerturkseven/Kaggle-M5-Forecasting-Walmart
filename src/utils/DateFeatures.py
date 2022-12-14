import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from typing import List, Optional


class DateFeatures:
    def __init__(
        self,
        *,
        input_col: str,
        features: List[str],
        output_cols: Optional[List[str]] = None,
    ) -> None:
        self.input_col = input_col
        self.features = features
        self.output_cols = output_cols

    @property
    def features(self):
        return self.features_

    @features.setter
    def features(self, value):
        supported_features = [
            "day_of_week",
            "day_of_year",
            "day_of_month",
            "week_of_year",
            "week_of_month",
            "month",
            "quarter",
            "year",
        ]
        if all([v in supported_features for v in value]):
            self.features_ = value
        else:
            raise ValueError(
                f"Invalid date frequency, supported {', '.join(supported_features)}"
            )

    @property
    def output_cols(self):
        return self.output_cols_

    @output_cols.setter
    def output_cols(self, value):
        if value is None:
            self.output_cols_ = self.features
        else:
            self.output_cols_ = value

    def transform(self, df):
        input_col = self.input_col
        output_cols = self.output_cols
        features = self.features
        for feature, output_col in zip(features, output_cols):
            if feature == "day_of_week":
                df = df.withColumn(output_col, F.dayofweek(F.col(input_col)))
            if feature == "day_of_year":
                df = df.withColumn(output_col, F.dayofyear(F.col(input_col)))
            if feature == "day_of_month":
                df = df.withColumn(output_col, F.dayofmonth(F.col(input_col)))
            if feature == "week_of_year":
                df = df.withColumn(output_col, F.weekofyear(F.col(input_col)))
            if feature == "week_of_month":
                df = df.withColumn(
                    output_col,
                    F.date_format(F.col(input_col), "W").cast("integer"),
                )
            if feature == "month":
                df = df.withColumn("month", F.month(F.col(input_col)))
            if feature == "quarter":
                df = df.withColumn("quarter", F.quarter(F.col(input_col)))
            if feature == "year":
                df = df.withColumn("year", F.year(F.col(input_col)))
        return df
