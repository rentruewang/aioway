# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.spaces import Attr, AttrDict

__all__ = ["Space", "TensorSpace", "TdictSpace", "SpaceList", "space_dcls"]


@typing.dataclass_transform(frozen_default=True)
def space_dcls[T](cls: type[Space[T]]):
    """
    The dataclass decorator for all `Space` subclasses.
    It's defined here as a standalone function
    s.t. you do not need to repeat dataclass configs for subclasses.
    """

    return dcls.dataclass(frozen=True, slots=True)(cls)


@space_dcls
class Space[T = typing.Any](abc.ABC):
    """
    The base class for tags. A tag describes a `torch.Tensor`,
    and is piggybacked onto the tensor to pass around `torch` APIs.

    It is also a filtering system.

    Each tag type (subclass) defines a tag name `cls.TAG`,
    corresponding to the attribute name it is set on the tensor.

    This means each tensor would have 1 tag of the same type,
    multiple tags would have multiple types. Therefore, an instance of tag
    is expected to fully describe a tensor in that tag's expertise.
    For example, a tag about batch dimensino should describe all dimensions at once,
    because that tag type can only have a singleton on each tensor.
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


@space_dcls
class TensorSpace(Space[torch.Tensor]):
    "A `Space` that enforces constraints on a `torch.Tensor`."

    @typing.override
    def contains(self, value: torch.Tensor, /) -> bool:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{type(value)} is not a `torch.Tensor`.")

        attr = Attr.parse(value)

        try:
            self._check_attr(attr)
            self._check_data(value)
        except ValueError:
            return False
        else:
            return True

    def _check_attr(self, attr: Attr, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tensor with `attr`.
        """

    def _check_data(self, tensor: torch.Tensor, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tensor`.
        """


@space_dcls
class TdictSpace(Space[td.TensorDict]):
    "A `Space` that checks a `td.TensorDict`."

    @typing.override
    def contains(self, value: td.TensorDict, /) -> bool:
        if not isinstance(value, td.TensorDict):
            raise TypeError(f"{type(value)} is not a `td.TensorDict`.")

        attrs = AttrDict.parse(value)

        try:
            self._check_attrs(attrs)
            self._check_data(value)
        except ValueError:
            return False
        else:
            return True

    def _check_attrs(self, attrs: AttrDict, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tdict with `attrs`.
        """

    def _check_data(self, tdict: td.TensorDict, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tdict`.
        """


@space_dcls
class SpaceList[T = typing.Any](Space[T]):
    """
    The tags stored on an object can be extracted into a `TagDict`,
    which supports both fast lookup and conveniently unpacks to only `Space`s.
    """

    spaces: cabc.Sequence[Space[T]] = ()
    "The spaces that this space list contains."

    def contains(self, item: T) -> bool:
        return all(item in space for space in self.spaces)

    def __iter__(self) -> cabc.Iterator[Space[T]]:
        yield from self.spaces

    def __len__(self) -> int:
        return len(self.spaces)

    def __getitem__(self, idx: int, /) -> Space[T]:
        return self.spaces[idx]
