import pyspark.sql.functions as F
from pyspark.sql.window import Window
import itertools
from pyspark.sql import DataFrame
from typing import List, Optional


class TargetEncoder:
    def __init__(
        self,
        *,
        value_col: str,
        date_col: str,
        rolling_windows: List[int],
        lags: List[int],
        functions: List[str],
        date_frequency: str,
        output_col_prefix: str,
        partition_cols: Optional[List[str]] = None,
    ) -> None:
        self.partition_cols = [] if partition_cols == None else partition_cols
        self.value_col = value_col
        self.date_col = date_col
        self.rolling_windows = rolling_windows
        self.lags = lags
        self.functions = functions
        self.date_frequency = date_frequency
        self.output_col_prefix = output_col_prefix

    @property
    def date_frequency(self):
        return self.date_frequency_

    @date_frequency.setter
    def date_frequency(self, value):
        supported_freq = ["day", "week", "month"]
        if value in supported_freq:
            self.date_frequency_ = value
        else:
            raise ValueError(
                f"Invalid date frequency, supported {', '.join(supported_freq)}"
            )

    def _period_diff(self, df):
        data_frequency = self.date_frequency
        date_col = self.date_col
        df = df.crossJoin(df.select(F.min(date_col).alias("min_period")))
        if data_frequency == "day":
            df = df.withColumn(
                "period_number", F.datediff(F.col(date_col), "min_period")
            )
        if data_frequency == "week":
            df = df.withColumn(
                "period_number", F.datediff(F.col(date_col), "min_period") / 7
            )
        if data_frequency == "month":
            df = df.withColumn(
                "period_number", F.months_between(F.col(date_col), "min_period")
            )
        return df.drop("min_period")

    def transform(self, df: DataFrame) -> DataFrame:
        partition_cols = self.partition_cols
        value_col = self.value_col
        rolling_windows = self.rolling_windows
        lags = self.lags
        functions = self.functions
        output_col_prefix = self.output_col_prefix
        df = self._period_diff(df)
        iter_list = itertools.product(rolling_windows, lags, functions)
        for rolling_window, lag, func in iter_list:
            w = (
                Window.partitionBy(partition_cols)
                .orderBy("period_number")
                .rangeBetween(-(lag + rolling_window - 1), -lag)
            )
            df = df.withColumn(
                f"{output_col_prefix}_window_{rolling_window}_lag_{lag}_{func}",
                F.expr(f"{func}({value_col})").over(w),
            )
        return df.drop("period_number")
