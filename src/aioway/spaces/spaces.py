# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Space` interface."

import abc
import dataclasses as dcls
import typing

import torch
from torchrl.data import tensor_specs as tspecs

__all__ = ["Space", "SpaceLike", "SpaceCompat", "TSpecSpace"]

type SpaceCompat = Space | tspecs.TensorSpec | SpaceLike
"""
Types compatible with `Space`.
"""


@typing.runtime_checkable
class SpaceLike(typing.Protocol):
    """
    The objects that can be casted to `Space` defines a `__space__` method.

    For convenience, allows a `TensorSpec` to be returned,
    will automatically be wrapped in an `TSpecSpace`.
    """

    def __space__(self) -> Space | tspecs.TensorSpec: ...


@dcls.dataclass(frozen=True)
class Space[T = typing.Any](abc.ABC):
    """
    `Space` acts as the types of data in `aioway`.

    It also acts as a filter in compiling the modules.
    """

    @abc.abstractmethod
    def __contains__(self, obj: T, /) -> bool:
        """
        Check if the object is in the current `Space`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, *shape: int) -> T:
        "Each shape supports sampling."

        raise NotImplementedError


def as_space(space: SpaceCompat, /) -> Space:
    """
    Convert an object into a `Space`. If attempts fail, a `TypeError` is raised.

    The argument `space` can either be one of the 3 types:

    1. `Space`. No conversion is made.
    2. `TensorSpec`. Would be wrapped in a `TensorSpecSpace`.
    3. `SpaceLike`. Types that define `__space__` (convert to `Space` or `TensorSpec`).
    """

    if isinstance(space, Space):
        return space

    if isinstance(space, tspecs.TensorSpec):
        return TSpecSpace(space)

    if isinstance(space, SpaceLike):
        return as_space(space.__space__())

    raise TypeError(f"Do not know how to handle {space=}.")


@typing.final
@dcls.dataclass(frozen=True)
class TSpecSpace[S: tspecs.TensorSpec = tspecs.TensorSpec](Space):
    """
    The `Space` that contains a `TensorSpec` from `torchrl`.

    This class exists because we want to maximally leverage `TensorSpec`,
    but having `Space | TensorSpec` is awkward because of 2 sets of APIs.
    """

    spec: S
    """
    The spec that the `Space` wraps.
    """

    @typing.override
    def __contains__(self, obj, /) -> bool:
        return self.spec.is_in(obj)

    @typing.override
    def sample(self, *shapes: int):
        return self.spec.sample(torch.Size(shapes))

    def cast_spec_type[T](self, spec_type: type[T], /) -> T | None:
        """
        Cast `self.spec` to the given `spec_type`.

        If the type cast failed, return `None`. Else return the instance.
        """

        if isinstance(self.spec, spec_type):
            return self.spec
        else:
            return None

    @property
    def ndim(self) -> int:
        return self.spec.ndim

    @property
    def shape(self) -> torch.Size:
        return self.spec.shape

    @property
    def device(self) -> torch.device | None:
        return self.spec.device

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype
