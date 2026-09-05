# Copyright (c) AIoWay Authors - All Rights Reserved

"The `TSpec` interface."

import abc
import functools
import typing

import tensordict as td
import torch
from torchrl.data import tensor_specs as tspecs

__all__ = [
    "TSpec",
    "TSpecLike",
    "TSpecCompat",
    "as_tspec",
    "is_tspec_like",
    "is_tspec_subtype",
]


type TSpecLike = TSpec | tspecs.TensorSpec | TSpecCompat
"""
Types compatible with `TSpec`.
"""


@typing.runtime_checkable
class TSpec(typing.Protocol):
    """
    `TSpec` is essentially a protocol that mimicks `TensorSpec`,
    but only contains the most important functionalities,
    s.t. our custom implementation won't be too burdened.

    Note:
        The methods are marked as `abc.abstractmethod` and `raise NotImplementedError`,
        despite this being a `Protocol`, because this is meant to be subclassed.
    """

    @abc.abstractmethod
    def is_in(self, obj, /) -> bool:
        raise NotImplementedError

    def contains(self, obj, /) -> bool:
        return self.is_in(obj)

    def assert_is_in(self, obj, /) -> None:
        assert self.is_in(obj)

    @property
    @abc.abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @abc.abstractmethod
    def rand(self, shape: torch.Size, /) -> torch.Tensor | td.TensorDictBase:
        raise NotImplementedError

    def sample(self, shape: torch.Size, /) -> torch.Tensor | td.TensorDictBase:
        return self.rand(shape)


@typing.runtime_checkable
class TSpecCompat(typing.Protocol):
    """
    The objects that can be casted to `TSpec` defines a `__tspec__` method.
    """

    def __tspec__(self) -> TSpec: ...


def is_tspec_subtype(cls: type) -> typing.TypeIs[type[TSpec]]:
    "Check if `cls` is a subclass of `TSpecLike`."
    return issubclass(cls, tspecs.TensorSpec) or _inherits_from_tspec(cls)


def is_tspec_like(spec) -> typing.TypeIs[TSpec]:
    "Check if `cls` is an instance of `TSpecLike`."
    return isinstance(spec, tspecs.TensorSpec | TSpec)


def as_tspec(spec: TSpecLike, /) -> TSpec:
    """
    Convert an object into a `TSpec`. If attempts fail, a `TypeError` is raised.

    The argument `spec` can either be one of the 3 types:

    1. `TSpec`. No conversion is made (this includes `TensorSpec`).
    2. `TSpecLike`. Types that define `__tspec__` (convert to `TSpec`).
    """

    if is_tspec_like(spec):
        return spec

    if isinstance(spec, TSpecCompat):
        return as_tspec(spec.__tspec__())

    raise TypeError(f"Do not know how to handle {spec=}.")


def _inherits_from_tspec(cls: type):
    bases = _find_bases(cls)
    return TSpec in bases


def _visit_bases(cls: type, seen: set[type]) -> None:
    if cls in seen:
        return

    seen.add(cls)

    for sub in cls.__bases__:
        _visit_bases(sub, seen)


@functools.cache
def _find_bases(cls: type) -> frozenset[type]:
    seen: set[type] = set()
    _visit_bases(cls, seen)
    return frozenset(seen)
