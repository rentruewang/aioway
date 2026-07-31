# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import typing

from aioway._api import public_api

__all__ = ["Space", "SpaceLike", "AnySpace", "space_dcls"]


@public_api
@typing.dataclass_transform(frozen_default=True)
def space_dcls[T](cls: type[Space[T]]):
    """
    The dataclass decorator for all `Space` subclasses.
    It's defined here as a standalone function
    s.t. you do not need to repeat dataclass configs for subclasses.
    """

    return dcls.dataclass(frozen=True, slots=True)(cls)


@public_api
@typing.runtime_checkable
class SpaceLike[T](typing.Protocol):
    def __space__(self) -> Space[T]: ...


@public_api
@space_dcls
class Space[T = typing.Any](abc.ABC):
    """
    The base class for spaces. A space describes an (batched) input or output,
    and is inspired by `gymnasium`'s `Space` class.

    It is also a filtering system.
    """

    def __contains__(self, value: T, /) -> bool:
        return self.contains(value)

    @abc.abstractmethod
    def contains(self, value: T, /) -> bool:
        """
        Perform some checks on the value you are going to attach on.
        If the tests pass, return `True`, else return `False`.
        """

        raise NotImplementedError


@public_api
@space_dcls
class AnySpace(Space):
    """
    A `Space` that imposes no constraints.
    """

    @typing.override
    def contains(self, value):
        return True
