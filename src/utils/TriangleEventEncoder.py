import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame
from typing import List, Optional


class TriangleEventEncoder:
    def __init__(
        self,
        *,
        partition_cols: List[str],
        input_cols: List[str],
        date_col: str,
        window: int,
        date_frequency: str,
        output_cols: Optional[List[str]] = None,
    ) -> None:
        self.partition_cols = partition_cols
        self.input_cols = input_cols
        self.date_col = date_col
        self.window = window
        self.date_frequency = date_frequency
        self.output_cols = output_cols

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

    @property
    def output_cols(self):
        return self.output_cols_

    @output_cols.setter
    def output_cols(self, value):
        if value is None:
            self.output_cols_ = self.input_cols
        else:
            self.output_cols_ = value

    def _period_diff(self, first_date, second_date):
        data_frequency = self.date_frequency
        if data_frequency == "day":
            return F.datediff(first_date, second_date)
        if data_frequency == "week":
            return F.datediff(first_date, second_date) / 7
        if data_frequency == "month":
            return F.months_between(first_date, second_date)

    def transform(self, df):
        partition_cols = self.partition_cols
        input_cols = self.input_cols
        output_cols = self.output_cols
        date_col = self.date_col
        window = self.window
        previous_rows = (
            Window.partitionBy(partition_cols)
            .orderBy(date_col)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
        following_rows = (
            Window.partitionBy(partition_cols)
            .orderBy(date_col)
            .rowsBetween(0, Window.unboundedFollowing)
        )
        for input_col, output_col in zip(input_cols, output_cols):
            df = (
                df.withColumn(
                    "last_event_date",
                    F.last(
                        F.when(F.col(input_col) == 1, F.col(date_col)), ignorenulls=True
                    ).over(previous_rows),
                )
                .withColumn(
                    "next_event_date",
                    F.first(
                        F.when(F.col(input_col) == 1, F.col(date_col)), ignorenulls=True
                    ).over(following_rows),
                )
                .withColumn(
                    "periods_to_event", self._period_diff("next_event_date", date_col)
                )
                .withColumn(
                    "periods_after_event",
                    self._period_diff("last_event_date", date_col),
                )
                .withColumn(
                    "periods_to_event",
                    F.when(F.col("periods_to_event") > window, None).otherwise(
                        F.col("periods_to_event")
                    ),
                )
                .withColumn(
                    "periods_after_event",
                    F.when(F.col("periods_after_event") < -window, None).otherwise(
                        F.col("periods_after_event")
                    ),
                )
                .withColumn(
                    output_col, F.greatest("periods_to_event", "periods_after_event")
                )
                .drop(
                    "last_event_date",
                    "next_event_date",
                    "periods_to_event",
                    "periods_after_event",
                )
            )
        return df
