from pyspark.sql import DataFrame
from typing import List


class JoinDataFrame:
    def __init__(self, *, df: DataFrame, on: List[str], how: str) -> None:
        self.df = df
        self.on = on
        self.how = how

    def transform(self, df: DataFrame) -> DataFrame:
        return df.join(self.df, on=self.on, how=self.how)
