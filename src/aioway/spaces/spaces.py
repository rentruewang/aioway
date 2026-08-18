# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Space` interface."

import abc
import dataclasses as dcls
import typing

from torchrl.data import tensor_specs as tspecs

__all__ = ["Space", "TensorSpecSpace"]


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


@typing.final
@dcls.dataclass(frozen=True)
class TensorSpecSpace[S: tspecs.TensorSpec = tspecs.TensorSpec](Space):
    """
    The `Space` that contains a `TensorSpec` from `torchrl`.
    """

    spec: S
    """
    The spec that the `Space` wraps.
    """

    @typing.override
    def __contains__(self, obj, /) -> bool:
        return self.spec.is_in(obj)
