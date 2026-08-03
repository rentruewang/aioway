# Copyright (c) AIoWay Authors - All Rights Reserved

"The executor for the relational algebra API."

import abc
import contextlib as ctxl
import copy
import typing
from collections import abc as cabc

from aioway._api import public_api
from aioway._torch import torch_fake_mode
from aioway._utils import decomp_dcls_members

from .nodes import GraphNode

if typing.TYPE_CHECKING:
    from aioway.relalg import ExecIter

__all__ = ["Exec", "sample_mode", "current_sample_mode"]

_sample_mode: bool = False
"Whether or not `Exec` is using fake data."


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


@public_api
class Exec[T](cabc.Iterable[T], GraphNode["Exec"], abc.ABC):
    """
    It produces iterators that computes the desired batch, represented by the node.

    It acts as a node in a DAG (that supports multiple fan-outs), that is,
    repeated `next` calls in the same pass would be cached.

    `Exec` is the node that would be evaluated during run time.
    It will output `torch.Tensor`, or a container that makes up of them.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __iter__(self) -> ExecIter[T]:
        from .iters import CacheExecIter, SampleExecIter

        # If `sample_mode` is on, use the `.sample()` method.
        if _sample_mode:
            return SampleExecIter(self)

        # Else yield from the `.iterate()` method.
        else:
            return CacheExecIter(self)

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
