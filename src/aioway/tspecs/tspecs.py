# Copyright (c) AIoWay Authors - All Rights Reserved

"The `TSpec` interface."

import abc
import typing

import tensordict as td
import torch

__all__ = ["TSpec", "TSpecLike", "TSpecCompat", "as_tspec", "is_tspec_like"]


type TSpecLike = TSpec | TSpecCompat
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

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

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


def is_tspec_like(spec: typing.Any, /) -> typing.TypeIs[TSpecLike]:
    """
    Check if item is like a `TSpec`.
    """

    try:
        _ = as_tspec(spec)
    except TypeError:
        return False
    else:
        return True


def as_tspec(spec: TSpecLike, /) -> TSpec:
    """
    Convert an object into a `TSpec`. If attempts fail, a `TypeError` is raised.

    The argument `spec` can either be one of the 3 types:

    1. `TSpec`. No conversion is made (this includes `TensorSpec`).
    2. `TSpecLike`. Types that define `__tspec__` (convert to `TSpec`).
    """

    if isinstance(spec, TSpec):
        return spec

    if isinstance(spec, TSpecCompat):
        return as_tspec(spec.__tspec__())

    raise TypeError(f"Do not know how to handle {spec=}.")
