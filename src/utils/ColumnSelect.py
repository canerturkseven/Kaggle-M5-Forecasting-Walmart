from pyspark.sql import DataFrame
from typing import List


class ColumnSelect:
    def __init__(self, cols: List[str]) -> None:
        self.cols = cols

    def transform(self, df: DataFrame) -> DataFrame:
        return df.select(self.cols)
