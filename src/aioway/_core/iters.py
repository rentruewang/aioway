# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator class."

import abc
import copy
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import decomp_dcls_members, torch_fake_mode
from aioway.spaces import Attr, Shape

from .nodes import GraphNode, node_dcls

if typing.TYPE_CHECKING:
    from .procs import IterProc

__all__ = ["Iter", "TensorIter", "TdictIter", "ListIter", "BoundedIter"]


class Iter[T](cabc.Iterable[T], GraphNode, abc.ABC):
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
        from .procs import IterProc

        return IterProc(self)

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
        e.g. switching on real mode after configuring the `HopDag` in fake mode.

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
        from aioway.hop.views import ColumnViewIter

        return ColumnViewIter(self, col)

    def select(self, *cols: str) -> TdictIter:
        from aioway.hop.views import ProjectIter

        return ProjectIter(self, subset=list(cols))


@node_dcls
class ListIter[T = typing.Any](Iter[cabc.Sequence[T]]):
    "A convenient list of `Iter`s, using a pull strategy to pull in the data when called."

    hops: cabc.Sequence[Iter[T]]
    """
    The hop list that this `HopList` represents.
    """

    def __repr__(self) -> str:
        return repr(self.hops)

    def __len__(self) -> int:
        return len(self.hops)

    @typing.overload
    def __getitem__(self, key: int) -> Iter[T]: ...

    @typing.overload
    def __getitem__(self, key: slice) -> typing.Self: ...

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.hops[key]

        if isinstance(key, slice):
            return type(self)(self.hops[key])

        raise TypeError(f"Don't know how to handle {type(key)=}.")

    @typing.override
    def iterate(self):
        for hops in zip(*self.hops):
            yield hops


@node_dcls
class BoundedIter(TdictIter, abc.ABC):
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
