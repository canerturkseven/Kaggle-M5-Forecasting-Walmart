import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from typing import List, Optional


class ConvertToDate:
    def __init__(
        self,
        *,
        input_cols: List[str],
        date_format: str,
        output_cols: Optional[List[str]] = None
    ) -> None:
        self.input_cols = input_cols
        self.date_format = date_format
        self.output_cols = output_cols

    @property
    def output_cols(self):
        return self.output_cols_

    @output_cols.setter
    def output_cols(self, value):
        if value is None:
            self.output_cols_ = self.input_cols
        else:
            self.output_cols_ = value

    def transform(self, df: DataFrame) -> DataFrame:
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            df = df.withColumn(
                output_col, F.to_date(F.col(input_col), self.date_format)
            )
        return df
