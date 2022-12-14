#%%
from pyspark.sql import SparkSession
from utils.MeltDataFrame import MeltDataFrame
from utils.JoinDataFrame import JoinDataFrame
from utils.ColumnSelect import ColumnSelect
from utils.DateFeatures import DateFeatures
from utils.CreateFutureSet import CreateFutureSet
from utils.ConvertToDate import ConvertToDate
from utils.TargetEncoder import TargetEncoder
from utils.LocalCheckpointer import LocalCheckpointer
from utils.ColumnType import ColumnType
from utils.TriangleEventEncoder import TriangleEventEncoder


def calendar_pipeline(calendar):
    column_select = ColumnSelect(
        cols=[
            "d",
            "date",
            "Snap_CA",
            "Snap_TX",
            "Snap_WI",
        ],
    )
    convert_date = ConvertToDate(
        input_cols=["date"],
        date_format="yyyy-MM-dd",
    )
    local_checkpoint = LocalCheckpointer(eager=True)
    calendar_pipeline = [column_select, convert_date, local_checkpoint]
    for i in calendar_pipeline:
        calendar = i.transform(calendar)
    return calendar


def actual_pipeline(actual, calendar):
    melt_df = MeltDataFrame(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ],
        value_vars=[f"d_{i}" for i in range(1, 1942)],
        var_name="d",
        value_name="sales",
    )
    cast_to_float = ColumnType(
        input_cols=["sales"],
        value="float",
    )
    join_calendar = JoinDataFrame(
        df=calendar,
        on="d",
        how="left",
    )
    future_df = CreateFutureSet(
        group_cols=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ],
        date_col="date",
        n_periods=28,
        date_frequency="day",
    )
    rolling_mean_std = TargetEncoder(
        partition_cols=["item_id", "store_id"],
        rolling_windows=[7, 15, 30, 90, 180],
        lags=[7, 14, 21, 28],
        date_frequency="day",
        functions=["mean", "std"],
        value_col="sales",
        date_col="date",
        output_col_prefix="item_store",
    )
    local_checkpoint = LocalCheckpointer(eager=True)
    date_features = DateFeatures(
        input_col="date",
        features=[
            "day_of_week",
            "day_of_month",
            "week_of_year",
            "week_of_month",
            "month",
            "year",
        ],
    )
    pipeline = [
        melt_df,
        cast_to_float,
        join_calendar,
        future_df,
        local_checkpoint,
        rolling_mean_std,
        date_features,
    ]
    for i in pipeline:
        actual = i.transform(actual)
    return actual


if __name__ == "__main__":
    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.driver.memory", "30g")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.shuffle.partitions", 2)
    spark.conf.set("spark.driver.maxResultSize", "20g")
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    spark.sparkContext.setCheckpointDir("/checkpoint")

    actual = spark.read.csv("../input/sample.csv", header=True)
    calendar = spark.read.csv("../input/calendar.csv", header=True)

    calendar = calendar_pipeline(calendar)
    df = actual_pipeline(actual, calendar)
    df.write.parquet("../output/feature")

#%%
