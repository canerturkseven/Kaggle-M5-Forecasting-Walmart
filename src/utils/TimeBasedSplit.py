import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class TimeBasedSplit:
    def __init__(
        self,
        *,
        date_frequency,
        date_col,
        test_size=1,
        max_train_size=None,
        gap=0,
        n_splits=5,
        end_offset=0,
    ):
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.n_splits = n_splits
        self.date_frequency = date_frequency
        self.gap = gap
        self.end_offset = end_offset
        self.date_col = date_col

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        if int(value) <= 0:
            raise ValueError(f"test_size must be positive, received {value} instead")
        else:
            self._test_size = value

    @property
    def max_train_size(self):
        return self._max_train_size

    @max_train_size.setter
    def max_train_size(self, value):
        if value:
            if int(value) <= 0:
                raise ValueError(
                    f"max_train_size must be positive, received {value} instead"
                )
        self._max_train_size = value

    @property
    def gap(self):
        return self._gap

    @gap.setter
    def gap(self, value):
        if int(value) < 0:
            raise ValueError(f"gap must be greater than zero, received {value} instead")
        else:
            self._gap = value

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, value):
        if int(value) <= 0:
            raise ValueError(f"n_splits must be positive, received {value} instead")
        else:
            self._n_splits = value

    @property
    def date_frequency(self):
        return self._date_frequency

    @date_frequency.setter
    def date_frequency(self, value):
        supported_date_frequency = [
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
        ]
        if not value in supported_date_frequency:
            raise ValueError(
                f"{value} is not supported as date frequency, "
                "supported frequencies are: {' '.join(supported_date_frequency)}"
            )
        else:
            self._date_frequency = value

    @property
    def end_offset(self):
        return self.end_offset_

    @end_offset.setter
    def end_offset(self, value):
        if int(value) < 0:
            raise ValueError(f"end_offset must be >= 0, received {value} instead")
        else:
            self.end_offset_ = value

    def _check_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

    def _check_date_col(self, df, date_col):
        if not date_col in df.columns:
            raise ValueError(f"{date_col} is not in the {df} columns")
        if not np.issubdtype(df[date_col].dtypes, np.datetime64):
            raise ValueError(f"{date_col} must be a date column")

    def _check_n_splits(self, df, date_col):
        max_date = df[date_col].max().to_pydatetime() - relativedelta(
            **{self.date_frequency: self.end_offset}
        )
        min_date = (
            max_date
            - relativedelta(**{self.date_frequency: self.gap})
            - (relativedelta(**{self.date_frequency: self.test_size}) * self.n_splits)
        )
        if df[df[date_col] <= min_date].empty:
            raise ValueError(
                f"Too many splits={self.n_splits} "
                f"with test_size={self.test_size} and gap={self.gap} "
                f"for the date sequence."
            )

    def split(self, df):
        gap = self.gap
        max_train_size = self.max_train_size
        date_frequency = self.date_frequency
        test_size = self.test_size
        n_splits = self.n_splits
        end_offset = self.end_offset
        date_col = self.date_col

        self._check_input(df)
        self._check_date_col(df, date_col)
        self._check_n_splits(df, date_col)

        df = df.reset_index(drop=True)
        max_date = df[date_col].max().to_pydatetime() - relativedelta(
            **{date_frequency: end_offset}
        )
        splits = []
        for i in range(n_splits):
            test_end = max_date - i * relativedelta(**{date_frequency: test_size})
            test_start = max_date - (i + 1) * relativedelta(
                **{date_frequency: test_size}
            )
            train_end = test_start - relativedelta(**{date_frequency: gap})
            test_condition = (df[date_col] > test_start) & (df[date_col] <= test_end)
            if self.max_train_size:
                train_start = train_end - relativedelta(
                    **{date_frequency: max_train_size}
                )
                train_condition = (df[date_col] > train_start) & (
                    df[date_col] <= train_end
                )
            else:
                train_condition = df[date_col] <= train_end
            splits.append(
                (
                    df[train_condition].index.tolist(),
                    df[test_condition].index.tolist(),
                )
            )
        return splits
