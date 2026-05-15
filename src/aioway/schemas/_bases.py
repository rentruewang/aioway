# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

__all__ = ["TorchAttrBase"]


class TorchAttrBase[T](abc.ABC):
    "The base class for the attributes."

    TYPE: typing.ClassVar[type]
    "The type used for checking at runtime."

    def __init__(self, data: T) -> None:
        self._data = data
        "The data that is stored."

        if not isinstance(self._data, self.TYPE):
            raise TypeError(f"Data received {data!r} is not of type {self.TYPE}.")

    @typing.final
    def __eq__(self, other: typing.Any, /) -> bool:
        if type(self) == type(other):
            assert isinstance(other, TorchAttrBase)
            return self._data == other._data

        if isinstance(other, self.TYPE):
            return self._data == other

        # Use `.parse` method, which can raise any error, signalling failure.
        try:
            result = self.parse(other)
        except Exception:
            return NotImplemented
        else:
            return self == result

    @typing.final
    def __repr__(self) -> str:
        "String representation with `aioway` qualifier."

        return f"aioway.{self!s}"

    @abc.abstractmethod
    def __str__(self) -> str:
        "Get the string representation."

        raise NotImplementedError

    @abc.abstractmethod
    def __getstate__(self) -> object:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @typing.final
    def torch(self) -> T:
        "Unwrap the underlying `torch` construct."
        return self._data

    @classmethod
    @abc.abstractmethod
    def parse(cls, value: typing.Any, /) -> typing.Self:
        """
        Attempt to parse the value given.
        If not handled, a `RuntimeError` should be raised..
        """

        raise NotImplementedError
