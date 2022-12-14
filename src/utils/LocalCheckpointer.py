from pyspark.sql import DataFrame


class LocalCheckpointer:
    def __init__(self, eager: bool) -> None:
        self.eager = eager

    def transform(self, df: DataFrame) -> DataFrame:
        return df.localCheckpoint(self.eager)
