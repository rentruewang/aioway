# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import pandas as pd

__all__ = ["MissingValues"]


class SourceLabel(typing.NamedTuple):
    "The x, y labels. Supports being decomposed."

    x: pd.DataFrame
    y: pd.DataFrame


class TrainTest(typing.NamedTuple):
    "The train and test labels. Supports being decomposed."

    train: SourceLabel
    test: SourceLabel


@dcls.dataclass(frozen=True)
class MissingValues:
    "The utilities around missing values."

    df: pd.DataFrame
    "The underlying dataframe."

    def partition(self) -> TrainTest:
        "Get the missing values partition."

        null_rows = self.null_rows
        null_cols = self.null_cols

        train_x = self.df.loc[~null_rows, ~null_cols]
        train_y = self.df.loc[~null_rows, null_cols]
        test_x = self.df.loc[null_rows, ~null_cols]
        test_y = self.df.loc[null_rows, null_cols]

        return TrainTest(SourceLabel(train_x, train_y), SourceLabel(test_x, test_y))

    def write_prediction(self, pred: pd.DataFrame):
        null_rows = self.null_rows
        null_cols = self.null_cols

        assert pred.shape == (len(null_rows), len(null_cols))
        self.df.loc[null_rows, null_cols] = pred

    @property
    def null_rows(self) -> pd.Series[bool]:
        return self.df.isnull().any(axis=1)

    @property
    def null_cols(self) -> pd.Series[bool]:
        return self.df.isnull().any(axis=1)
