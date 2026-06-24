# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator class."

import abc
import contextlib as ctxl
import copy
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import AnySet, decomp_dcls_members, decomp_replace, torch_fake_mode
from aioway.spaces import Attr, Shape

from .nodes import GraphNode, node_dcls

if typing.TYPE_CHECKING:
    from .procs import IterProc

__all__ = [
    "Iter",
    "TensorIter",
    "TdictIter",
    "StructIter",
    "ListIter",
    "IndexibleIter",
    "sample_mode",
    "current_sample_mode",
]

_sample_mode: bool = False
"Whether or not `Iter` is using fake data."


@ctxl.contextmanager
def sample_mode(to: bool = True) -> cabc.Generator[None]:
    """
    Set sample mode to the given value.
    """

    global _sample_mode
    before = _sample_mode
    _sample_mode = to
    try:
        yield
    finally:
        _sample_mode = before


def current_sample_mode():
    "Get the current sample mode."
    return _sample_mode


class Iter[T](cabc.Iterable[T], GraphNode["Iter"], abc.ABC):
    """
    The class that defines [h]igh level [op]erations.
    It produces iterators that computes the desired batch, represented by the node.

    `Iter` is the node that would be evaluated during run time.
    It will output `torch.Tensor`, or a container that makes up of them.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __iter__(self) -> IterProc[T]:
        # Every iteration should yield a new `Iterator`.
        from .procs import CacheIterProc, SampleIterProc

        # If `sample_mode` is on, use the `.sample()` method.
        if _sample_mode:
            return SampleIterProc(self)

        # Else yield from the `.iterate()` method.
        else:
            return CacheIterProc(self)

    @abc.abstractmethod
    def iterate(self) -> cabc.Iterator[T]:
        """
        The iteration logic.
        Should invoke dependencies' via `__iter__` methods (just use `for` loops).
        """

        raise NotImplementedError

    def sample(self) -> T:
        """
        Get a sample from the input. This method should be equivalent to `next(iter(self))`,
        but in fake mode (no data is actually retrieved).
        """

        with torch_fake_mode():
            return next(iter(self))

    def rebuild(self):
        """
        Rebuild the current `Iter`. This is useful when you are switching contexts,
        e.g. switching on real mode after configuring the `Iter` in fake mode.

        If `self._rebuild()` is not overwritten, defaults to shallow copying `self`.
        """

        copied = self._rebuild()
        assert copied is not self
        return copied

    def _rebuild(self) -> typing.Self:
        return copy.copy(self)

    @typing.override
    def deps(self) -> cabc.Iterator[Iter]:
        "Decompose `self`, get the immediate dependencies."

        for hop in decomp_dcls_members(self, Iter):
            yield hop

    @property
    def size(self) -> int:
        """
        The length of the current stream `TdictIter`.

        This should be defined for relational algebra purposes.
        """

        return NotImplemented

    @classmethod
    def deps_type(cls):
        return Iter


@node_dcls
class TensorIter(Iter[torch.Tensor], abc.ABC):
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


@node_dcls
class TdictIter(Iter[td.TensorDict], abc.ABC):
    """
    An iterator of batches of `td.TensorDict`.

    This is the core abstraction used by the relational algebra operators.
    """

    def column(self, col: str) -> TensorIter:
        from aioway.relalg.views import ColumnViewIter

        return ColumnViewIter(self, col)

    def select(self, *cols: str) -> TdictIter:
        from aioway.relalg.views import ProjectIter

        return ProjectIter(self, subset=list(cols))


@node_dcls
class StructIter(Iter[typing.Any]):
    """
    An `Iter` that transforms structures of `Iter`s to `Iter` of structures.

    E.g. `[Iter[int], Iter[int]] -> Iter[[int, int]]`.
    """

    struct: typing.Any
    """
    The structure that this `StructList` would decompose and compute the next iterator.
    """

    def __repr__(self) -> str:
        return repr(self.struct)

    @typing.override
    def iterate(self):
        from .procs import iter_cache_on

        struct_of_iter = decomp_replace(self.struct, _start_sub_iter)

        while True:
            try:
                with iter_cache_on():
                    yield decomp_replace(struct_of_iter, _next_sub_iter)
            except StopIteration:
                return


def _start_sub_iter(item: object) -> IterProc[typing.Any]:
    "Start the iterator and store the started iterators into an `AnySet`."

    if not isinstance(item, Iter):
        return NotImplemented

    return iter(item)


def _next_sub_iter(item: object):
    "Get the next item."

    from .procs import IterProc

    if not isinstance(item, IterProc):
        return NotImplemented

    return next(item)


@node_dcls
class ListIter[T = typing.Any](Iter[cabc.Sequence[T]]):
    "A convenient list of `Iter`s, using a pull strategy to pull in the data when called."

    seqs: cabc.Sequence[Iter[T]]
    """
    The hop list that this `ListIter` represents.
    """

    def __repr__(self) -> str:
        return repr(self.seqs)

    @typing.overload
    def __getitem__(self, key: int) -> Iter[T]: ...

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
class IndexibleIter(TdictIter, abc.ABC):
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
