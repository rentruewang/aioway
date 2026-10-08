# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Query` type for `Program`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np

from aioway._utils import IntArray
from aioway.ir.instrs import Instr

if typing.TYPE_CHECKING:
    from .progs import Program

__all__ = ["Query", "QuerySel"]


class Query[I: Instr = typing.Any](abc.ABC):
    @abc.abstractmethod
    def __call__(self, program: Program[I], /) -> QuerySel:
        raise NotImplementedError


@typing.final
@dcls.dataclass(frozen=True)
class QuerySel:
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

    def min(self) -> int:
        return self.arr.min().item()

    def max(self) -> int:
        return self.arr.max().item()

    @classmethod
    def from_list_int(cls, lst: list[int]) -> typing.Self:
        return cls(np.asarray(lst))
