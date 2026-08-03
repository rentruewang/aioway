# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator class."

import abc
import contextlib as ctxl
import copy
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._api import public_api
from aioway._utils import AnyDict, decomp_dcls_members, decomp_replace, torch_fake_mode
from aioway.spaces import Attr, Shape

from .nodes import GraphNode, node_dcls

__all__ = [
    "Exec",
    "TensorExec",
    "TdictExec",
    "StructExec",
    "ListExec",
    "IndexibleExec",
    "sample_mode",
    "current_sample_mode",
    "IterProc",
    "SampleIterProc",
    "CacheIterProc",
    "iter_cache",
    "iter_cache_on",
]

_sample_mode: bool = False
"Whether or not `Exec` is using fake data."


_iter_cache: AnyDict[Exec] | None = None
"The cache instance for `Exec`."


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


@ctxl.contextmanager
def iter_cache_on() -> cabc.Generator[AnyDict[Exec]]:
    """
    Turn on caching for `Exec`. The cache is re-used in nested `iter_cache_on` blocks.

    Returns:
        A context manager that when activates, intercept all `Exec.__next__` calls,
        and stores the outputs s.t. second `.__next__()` uses the previous rersult.
    """

    global _iter_cache

    if _iter_cache is not None:
        yield _iter_cache
        return

    _iter_cache = AnyDict[Exec](Exec)

    try:
        yield _iter_cache
    finally:
        _iter_cache = None


def iter_cache() -> AnyDict[Exec]:
    """
    The active cache for `Exec`. If there is no active session, raise `RuntimeError`.
    """

    if _iter_cache is None:
        raise RuntimeError("`iter_cache` can only be called in `iter_cache_on` scope.")

    return _iter_cache


@public_api
class Exec[T](cabc.Iterable[T], GraphNode["Exec"], abc.ABC):
    """
    The class that defines [h]igh level [op]erations.
    It produces iterators that computes the desired batch, represented by the node.

    `Exec` is the node that would be evaluated during run time.
    It will output `torch.Tensor`, or a container that makes up of them.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __iter__(self) -> IterProc[T]:
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
        Rebuild the current `Exec`. This is useful when you are switching contexts,
        e.g. switching on real mode after configuring the `Exec` in fake mode.

        If `self._rebuild()` is not overwritten, defaults to shallow copying `self`.
        """

        copied = self._rebuild()
        assert copied is not self
        return copied

    def _rebuild(self) -> typing.Self:
        return copy.copy(self)

    @typing.override
    def deps(self) -> cabc.Iterator[Exec]:
        "Decompose `self`, get the immediate dependencies."

        for hop in decomp_dcls_members(self, Exec):
            yield hop

    @property
    def size(self) -> int:
        """
        The length of the current stream `TdictExec`.

        This should be defined for relational algebra purposes.
        """

        return NotImplemented

    @classmethod
    def deps_type(cls):
        return Exec


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
        next_it = lambda it: (next(it) if isinstance(it, IterProc) else NotImplemented)

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


class IterProc[T = typing.Any](cabc.Iterator[T], abc.ABC):
    def __init__(self, iterable: Exec) -> None:
        self._iter = iterable

    @abc.abstractmethod
    def __next__(self) -> T:
        raise NotImplementedError


class SampleIterProc[T = typing.Any](IterProc[T]):
    """
    Calls the `Exec.sample` method, which returns a fake output.
    This is invoked when `sample_mode` is on.
    """

    def __init__(self, iterable: Exec) -> None:
        super().__init__(iterable)
        assert current_sample_mode()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # Set sample mode to false to avoid self recursion.
        with sample_mode(False), torch_fake_mode():
            return self._iter.sample()


class CacheIterProc[T = typing.Any](IterProc[T]):
    """
    Calls the `.iterate` method, and cache it if `iter_cache` is enabled.
    This allows `.iterate` to be generators (more elegant).
    """

    def __init__(self, iterable: Exec) -> None:
        super().__init__(iterable)

        self.__idx: int = 0
        self.__gen = iterable.iterate()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # If `StopIteration` is raised here, it's done.

        with iter_cache_on():
            answer = self.read()

        self.__idx += 1
        return answer

    def read(self) -> T:
        if _iter_cache is None:
            return next(self.__gen)

        elif self.iterator not in _iter_cache:
            _iter_cache[self.iterator] = next(self.__gen)

        result: typing.Any = _iter_cache[self.iterator]
        return result

    @property
    def idx(self) -> int:
        "Get the current iteration count."

        return self.__idx

    @property
    def started(self) -> bool:
        "Shortcut function to check if `self.idx == 0`."

        return self.idx != 0

    @property
    def iterator(self) -> Exec:
        return self._iter
