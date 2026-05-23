# Copyright (c) AIoWay Authors - All Rights Reserved

"`Dataset` is base class for all datasets."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.schemas import Attr, AttrDict

__all__ = ["Dataset", "DatasetColumnView", "DatasetSelectView", "DatasetViewTypes"]


class Dataset(abc.ABC):
    """
    A tabular type that acts like a table, and is the shared base class for `Frame` and `Stream`.

    A `Dataset` should support the following functions:

    1. `column(key: str, /) -> `.
        Getting the individual column.
    2. `select(*keys: str) -> typing.Self`.
        Getting a couple of columns should return the same `Table`.
    3. `keys() -> cabc.KeysView[str]`

    """

    @typing.overload
    def __getitem__(self, key: str, /) -> DatasetColumnView[typing.Self]: ...

    @typing.overload
    def __getitem__(self, key: list[str], /) -> DatasetSelectView[typing.Self]: ...

    def __getitem__(self, key, /):
        match key:
            case str():
                return self.column(key)
            case list() if all(isinstance(i, str) for i in key):
                return self.select(*key)

        raise TypeError(
            "The default implemenetation of `Dataset.__getitem__` "
            f"does not know how to handle {key=}. "
            "It only supports `key` of type `str` and `list[str]`."
        )

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrDict:
        "All datasets have the metadta `attrs` present."

        raise NotImplementedError

    @typing.final
    def keys(self) -> cabc.KeysView[str]:
        """
        A `cabc.KeysView` object.
        """

        return self.attrs.keys()

    def column(self, key: str) -> DatasetColumnView[typing.Self]:
        """
        Get the column from the `Dataset` object.
        A `KeyError` is raised if the column is not present.

        Essentially this is the `cabc.Mapping.__getitem__` method,
        but a normal method to simplify implementation.

        Args:
            key: The column name.

        Returns:
            The column instance.

        Raises:
            KeyError: If the column is not present.
        """

        col_type, _ = self.view_types()
        return col_type.from_column(self, key)

    def select(self, *keys: str) -> DatasetSelectView[typing.Self]:
        """
        Select multiple columns from the `Dataset` object.

        If a key is missing, a `KeyError` is raised.

        Returns:
            A `Dataset` that wraps the result.
        """

        _, select_type = self.view_types()
        return select_type.from_columns(self, *keys)

    @classmethod
    @abc.abstractmethod
    def view_types(cls) -> DatasetViewTypes[typing.Any]:
        """
        The type used to construct `.column`, `.select` views.

        The reason this is not a `typing.ClassVar` is purely technical,
        because `*SelectView`s often inherit from `typing.Self`,
        making it a circular dependency if it were a `typing.ClassVar`.
        """

        raise NotImplementedError


@dcls.dataclass(frozen=True)
class DatasetView[T: Dataset](abc.ABC):

    dset: T
    "The original dataset that would be used in the view."


@dcls.dataclass(frozen=True)
class DatasetColumnView[T: Dataset = Dataset](DatasetView[T], abc.ABC):
    col: str
    "The column to pick. Must be in the original table."

    def __post_init__(self) -> None:
        _assert_column_in_dataset(self.col, self.dset.attrs)

    @property
    @typing.final
    def attr(self) -> Attr:
        return self.dset.attrs[self.col]

    @classmethod
    @abc.abstractmethod
    def from_column(cls, dataset: T, /, column: str) -> typing.Self: ...


@dcls.dataclass(frozen=True)
class DatasetSelectView[T: Dataset = Dataset](Dataset, DatasetView[T], abc.ABC):
    """
    Perform a selection in the table.
    This is a `View`, which means creation is cheap, but you pay the price in runtime.
    """

    COLUMN_TYPE: typing.ClassVar[type[DatasetColumnView[T]]]
    "The column type associated with the current `DatasetSelectView`."

    cols: cabc.Sequence[str]
    "The columns to select. Should be in the original table."

    def __post_init__(self) -> None:
        for col in self.cols:
            _assert_column_in_dataset(col, self.dset.attrs)

    @property
    @typing.final
    def attrs(self) -> AttrDict:
        return self.dset.attrs.select(*self.cols)

    @typing.final
    def column(self, key: str):
        _assert_column_in_dataset(key, self.attrs)
        return self.COLUMN_TYPE.from_column(self.dset, column=key)

    @typing.final
    def select(self, *keys: str):
        for key in keys:
            _assert_column_in_dataset(key, self.attrs)

        return self.dset.select(*keys)

    @classmethod
    @abc.abstractmethod
    def from_columns(cls, dataset: T, /, *columns: str) -> typing.Self: ...


class DatasetViewTypes[T: Dataset](typing.NamedTuple):
    "The view types."

    column: type[DatasetColumnView[T]]
    "The type used to construct `.column` views."

    select: type[DatasetSelectView[T]]
    "The type used to construct `.select` views."


def _assert_column_in_dataset(col: str, attrs: AttrDict) -> None:
    if col in attrs.keys():
        return

    raise ValueError(f"Column {col} is not present in the original dataset: {attrs=}")
