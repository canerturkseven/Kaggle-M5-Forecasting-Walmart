#%%
import pandas as pd
import itertools

df = pd.read_csv("sample.csv")
calendarrr = pd.read_csv("calendar.csv")


#%%
x = df.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_name="sales",
    var_name="d",
)
x = x.merge(calendar, on="d", how="left")
x["date"] = pd.to_datetime(x["date"], format="%Y-%m-%d")
#%%

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.driver.memory", "30g")
    .getOrCreate()
)
spark.conf.set("spark.sql.shuffle.partitions", 1)
spark.conf.set("spark.sql.execution.arrow.enabled", "false")
spark.sparkContext.setCheckpointDir("/checkpoint")
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

mySchema = StructType(
    [
        StructField("item_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("date", DateType(), True),
        StructField("sales", IntegerType(), True),
    ]
)
y = spark.createDataFrame(x[["item_id", "store_id", "date", "sales"]], schema=mySchema)
y.show()

from TargetEncoder import TargetEncoder

model = TargetEncoder(
    partition_cols=["item_id", "store_id"],
    value_col="sales",
    rolling_windows=[3],
    lags=[1],
    functions=["mean"],
    date_frequency="day",
    date_col="date",
    output_col_prefix="item_store",
)


model.transform(y).show()


# %%
class TargetEncoder:
    def __init__(
        self,
        partition_by,
        windows,
        lags,
        value_col,
        date_col,
        date_frequency,
        functions,
    ):
        self.partition_by = partition_by
        self.windows = windows
        self.lags = lags
        self.value_col = value_col
        self.date_col = date_col
        self.date_frequency = date_frequency
        self.functions = functions

    def _period_diff(self, df):
        data_frequency = self.date_frequency
        date_col = self.date_col
        date_period = df[date_col].dt.to_period(data_frequency)
        df["n_period"] = (date_period - date_period.min()).apply(lambda x: x.n)
        return df

    @staticmethod
    def _range_between(df, range_col, start, end, value_col, function, output_col):
        for i in range(len(df)):
            n_period = df.loc[df.index[i], range_col]
            mask = df[range_col].between(
                n_period + start, n_period + end, inclusive="right"
            )
            df.loc[df.index[i], output_col] = df.loc[mask, value_col].agg(function)
        return df

    def transform(self, df):
        windows = self.windows
        lags = self.lags
        functions = self.functions
        value_col = self.value_col
        df = self._period_diff(df)
        iter_list = itertools.product(windows, lags, functions)
        for window, lag, function in iter_list:
            output_col = (
                f"{'_'.join(self.partition_by)}_window_{window}_lag_{lag}_{function}"
            )
            df = df.groupby(self.partition_by, group_keys=False).apply(
                self._range_between,
                range_col="n_period",
                start=-(lag + window),
                end=-lag,
                value_col=value_col,
                function=function,
                output_col=output_col,
            )
        return df


model = TargetEncoder(
    partition_by=["item_id", "store_id"],
    windows=[3],
    lags=[1],
    value_col="sales",
    date_col="date",
    date_frequency="D",
    functions=["mean"],
)
r = model.transform(x)
#%%


# %%
