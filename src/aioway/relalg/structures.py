# Copyright (c) AIoWay Authors - All Rights Reserved

"Pack / unpack common structures in `Exec`s."

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._api import public_api
from aioway._torch import Attr, Shape
from aioway._utils import decomp_replace

from .execs import Exec
from .iters import ExecIter
from .nodes import node_dcls

__all__ = [
    "TensorExec",
    "TdictExec",
    "StructExec",
    "ListExec",
    "IndexibleExec",
]


@public_api
@node_dcls
class TensorExec(Exec[torch.Tensor], abc.ABC):
    """
    An iterator of batches of `torch.Tensor`.
    """

    @property
    def attr(self) -> Attr:
        return Attr.parse(self.sample())

    @property
    def shape(self) -> Shape:
        return self.attr.shape

    @property
    def ndim(self) -> int:
        return self.attr.ndim


@public_api
@node_dcls
class TdictExec(Exec[td.TensorDict], abc.ABC):
    """
    An iterator of batches of `td.TensorDict`.

    This is the core abstraction used by the relational algebra operators.
    """

    def column(self, col: str) -> TensorExec:
        from aioway.relalg.views import ColumnViewExec

        return ColumnViewExec(self, col)

    def select(self, *cols: str) -> TdictExec:
        from aioway.relalg.views import ProjectExec

        return ProjectExec(self, subset=list(cols))


@public_api
@node_dcls
class StructExec(Exec[typing.Any]):
    """
    An `Exec` that transforms structures of `Exec`s to `Exec` of structures.

    E.g. `[Exec[int], Exec[int]] -> Exec[[int, int]]`.
    """

    struct: typing.Any
    """
    The structure that this `StructList` would decompose and compute the next iterator.
    """

    def __repr__(self) -> str:
        return repr(self.struct)

    @typing.override
    def iterate(self):
        # Start the iterator and store the started iterators into an `AnySet`.
        start_it = lambda it: (iter(it) if isinstance(it, Exec) else NotImplemented)

        # Get the next item.
        next_it = lambda it: (next(it) if isinstance(it, ExecIter) else NotImplemented)

        # Inside the structure, call `iter` on all `Exec`s.
        struct_of_iter = decomp_replace(self.struct, start_it)

        # Inside the structure, call `next` on all `IterProc`s, until `StopIteration`.
        while True:
            try:
                yield decomp_replace(struct_of_iter, next_it)
            except StopIteration:
                return


@public_api
@node_dcls
class ListExec[T = typing.Any](Exec[cabc.Sequence[T]]):
    "A convenient list of `Exec`s, using a pull strategy to pull in the data when called."

    seqs: cabc.Sequence[Exec[T]]
    """
    The hop list that this `ListExec` represents.
    """

    def __repr__(self) -> str:
        return repr(self.seqs)

    @typing.overload
    def __getitem__(self, key: int) -> Exec[T]: ...

    @typing.overload
    def __getitem__(self, key: slice) -> typing.Self: ...

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.seqs[key]

        if isinstance(key, slice):
            return type(self)(self.seqs[key])

        raise TypeError(f"Don't know how to handle {type(key)=}.")

    @typing.override
    def iterate(self):
        for hops in zip(*self.seqs):
            yield hops


@node_dcls
class IndexibleExec(TdictExec, abc.ABC):
    """
    A stream with `__len__` and `__getitem__`.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        "The number of batches saved in the current `Stream`."

        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key: int, /) -> td.TensorDict:
        """
        Get individual items. Does not support slice input.

        Args:
            idx: An integer. Must be in the range `[-len(self), len(self))`.

        Returns:
            The `td.TensorDict` batch.
        """

        raise NotImplementedError
