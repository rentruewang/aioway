# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Query` type for `Program`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np

from aioway._utils import IntArray, is_list_of
from aioway.ir.instrs import Instr

if typing.TYPE_CHECKING:
    from aioway.ir import Program

__all__ = ["Query", "OrderedIndex"]

type _OrderedIndexLike = OrderedIndex | list[int] | IntArray


class Query[I: Instr = typing.Any](abc.ABC):
    @abc.abstractmethod
    def __call__(self, program: Program[I], /) -> OrderedIndex:
        raise NotImplementedError


@typing.runtime_checkable
class QueryLike[I: Instr = typing.Any](typing.Protocol):
    def __call__(self, program: Program[I], /) -> _OrderedIndexLike: ...


def register_query_function[I: Instr](function: QueryLike[I]) -> Query[I]:
    """
    The decorator to convert a statless function returning
    """

    raise NotImplementedError


@typing.final
@dcls.dataclass(frozen=True)
class OrderedIndex:
    """
    The query selection type. Follows the proof by construction style:
    Basically an `IntArray` but with additional checks on init.
    """

    arr: IntArray
    "The underlying array."

    def __post_init__(self) -> None:
        if not self.arr.size:
            raise ValueError("Expected a list, got a scalar.")

        if self.arr.ndim != 1:
            raise ValueError("Expected 1D array.")

        if (self.arr < 0).any():
            raise ValueError("Negative indices not allowed.")

        if len(self.arr) != len(np.unique(self.arr)):
            raise ValueError("Not unique.")

        if (self.arr[:-1] > self.arr[1:]).any():
            raise ValueError("Not sorted.")

    def __len__(self) -> int:
        return len(self.arr)

    @typing.overload
    def __getitem__(self, idx: int) -> int: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int] | IntArray) -> typing.Self: ...

    def __getitem__(self, idx):
        result = self.arr[idx]

        if isinstance(idx, int):
            return result

        else:
            return type(self)(result)

    def __contains__(self, x) -> bool:
        if not isinstance(x, int):
            raise TypeError("Not integer.")

        idx = np.searchsorted(self.arr, x).item()
        return idx < self.arr.size and self.arr[idx] == x

    def __iter__(self) -> cabc.Iterator[int]:
        return iter(self.arr)

    def __array__(self) -> IntArray:
        return self.arr

    def tolist(self) -> list[int]:
        return self.arr.tolist()

    def min(self) -> int:
        return self.arr[0].item()

    def max(self) -> int:
        return self.arr[-1].item()

    @classmethod
    def from_list_int(cls, lst: list[int]) -> typing.Self:
        return cls(np.asarray(lst))

    @classmethod
    def from_like(cls, like: _OrderedIndexLike) -> typing.Self:
        if isinstance(like, cls):
            return like

        if isinstance(like, np.ndarray):
            if not np.isdtype(like.dtype, "integral"):
                raise TypeError("Non int numpy array found.")

            return cls(like)

        if is_list_of(int)(like):
            return cls.from_list_int(like)

        typing.assert_never(like)
