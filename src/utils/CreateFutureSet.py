import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from typing import List, Optional

class CreateFutureSet:

    def __init__(
        self,
        *,
        date_col : str,
        n_periods : int,
        date_frequency: str,
        group_cols : Optional[List[str]] = None,
        ) -> None:
        self.date_col = date_col
        self.n_periods = n_periods
        self.date_frequency = date_frequency
        self.group_cols = [] if group_cols is None else group_cols

    @property
    def date_frequency(self):
        return self.date_frequency_

    @date_frequency.setter
    def date_frequency(self, value):
        supported_freq =['day','week','month'] 
        if value in supported_freq:
            self.date_frequency_ = value
        else:
            raise ValueError(f"Invalid date frequency, supported {', '.join(supported_freq)}")
        
    def _period_add(self, date_col, n_period_col):
        if self.date_frequency == 'day':
            return F.expr(f"date_add({date_col}, {n_period_col})")
        if self.date_frequency == 'week':
            return F.expr(f"date_add({date_col}, 7*{n_period_col})")
        if self.date_frequency == 'month':
            return F.expr(f"add_months({date_col}, {n_period_col})")

    def transform(self, df: DataFrame) -> DataFrame:
        group_cols = self.group_cols
        n_periods = self.n_periods
        date_col = self.date_col
        future_periods = (
            df
            .select(F.max(date_col).alias('min_period'))
            .withColumn('n_period', F.array(*map(F.lit, range(1, n_periods+1))))
            .select('min_period', F.explode('n_period').alias('n_period'))
            .select(self._period_add('min_period', 'n_period').alias(date_col))
        )
        df_future = df.select(group_cols).distinct().crossJoin(future_periods)
        other_cols = [col for col in df.columns if col not in [*group_cols, date_col]]
        for col in other_cols:
            df_future = df_future.withColumn(col , F.lit(None))
        return df.unionByName(df_future)
        