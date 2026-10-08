# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

from torch.utils import _pytree as pyt

from aioway._utils import AnyDict, any_dict, find_common_base, is_list_of
from aioway.t import TList

__all__ = ["Instr", "InstrList"]


@typing.dataclass_transform(frozen_default=True, eq_default=False)
def instr_dcls[T](cls):
    """
    The dataclass transform wrapper for `Instr` classes.

    This also registeres the class to `pyt`.
    """

    cls = dcls.dataclass(frozen=True, eq=False, repr=False)(cls)
    pyt.register_dataclass(cls)
    return cls


# The base instruction class ====


@instr_dcls
class Instr(abc.ABC):
    "The instruction base class."

    def tree_map(self, func: cabc.Callable):
        """
        Call the `tree_map` function on the `Instr`.
        """

        return pyt.tree_map(func=func, tree=self)

    def tree_map_only(self, types: type | tuple[type, ...], func: cabc.Callable):
        """
        Call the `tree_map_only` function on the `Instr`.
        """

        return pyt.tree_map_only(types, func=func, tree=self)

    @functools.cached_property
    def inputs(self) -> TList:
        return self._inputs()

    @functools.cached_property
    def outputs(self) -> TList:
        return self._outputs()

    @abc.abstractmethod
    def _inputs(self) -> TList:
        "The list of tensors in the inputs."

        raise NotImplementedError

    @abc.abstractmethod
    def _outputs(self) -> TList:
        "The list of tensors in the outputs."

        raise NotImplementedError


# The instr list class ====


class InstrList[I: Instr = typing.Any]:
    "The list of `Instr`."

    def __init__(self, iterable: tuple[I, ...]) -> None:
        """
        Args:
            iterable: The iterable data.
        """

        self._seq = iterable

    def __len__(self) -> int:
        return len(self._seq)

    @typing.overload
    def __getitem__(self, idx: int) -> I: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int]) -> typing.Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._seq[idx]

        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])

        if is_list_of(int)(idx):
            return type(self)(tuple(self._seq[i] for i in idx))

        raise IndexError(idx)

    def __iter__(self) -> cabc.Iterator[I]:
        return iter(self._seq)

    def index(self, instr: I, /) -> int:
        return self._instr_to_step[instr]

    @functools.cached_property
    def _instr_to_step(self) -> AnyDict[I, int]:
        "Mapping from thunks to their indices."

        return any_dict(Instr, *((thunk, idx) for idx, thunk in enumerate(self._seq)))

    @property
    def base(self) -> type[Instr] | None:
        """
        Get the common base type. If sequence is empty, return `None`.
        """

        # Pass in `self` works, but `tuple` for better performance.
        return find_common_base(self._seq)

    @classmethod
    def build(cls, iterable: cabc.Iterable[I] | typing.Self) -> typing.Self:
        if isinstance(iterable, InstrList):
            result: typing.Any = iterable
            return result

        return cls(tuple(iterable))
