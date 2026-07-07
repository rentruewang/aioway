# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import typing

import tensordict as td
import torch

from aioway._api import public_api
from aioway._utils import is_fake_tensor, is_real_tensor, torch_fake_mode
from aioway.attrs import Attr, AttrDict

__all__ = ["Space", "AnySpace", "TensorSpace", "TdictSpace", "space_dcls"]


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

    def sample(self, batch_size: int = 1) -> T:
        with torch_fake_mode():
            return self._sample_n(batch_size)

    @abc.abstractmethod
    def _sample_n(self, n: int, /) -> T:
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

    @typing.override
    def _sample_n(self, batch_size: int):
        return object()


@public_api
@space_dcls
class TensorSpace(Space[torch.Tensor], abc.ABC):
    "A `Space` that enforces constraints on a `torch.Tensor`."

    @typing.override
    @typing.final
    def contains(self, tensor: torch.Tensor, /) -> bool:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{type(tensor)} is not a `torch.Tensor`.")

        attr = Attr.parse(tensor)

        try:
            self._check_attr(attr)

            # Only perform the data checks if all the tensor is real.
            if not is_fake_tensor(tensor):
                self._check_data(tensor)
        except ValueError:
            return False
        else:
            return True

    @abc.abstractmethod
    def _check_attr(self, attr: Attr, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tensor with `attr`.
        """

    @abc.abstractmethod
    def _check_data(self, tensor: torch.Tensor, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tensor`.
        """


@public_api
@space_dcls
class TdictSpace(Space[td.TensorDict]):
    "A `Space` that checks a `td.TensorDict`."

    @typing.override
    @typing.final
    def contains(self, tdict: td.TensorDict, /) -> bool:
        if not isinstance(tdict, td.TensorDict):
            raise TypeError(f"{type(tdict)} is not a `td.TensorDict`.")

        attrs = AttrDict.parse(tdict)

        try:
            self._check_attrs(attrs)

            # Only perform the data checks if all the values are real.
            if all(map(is_real_tensor, tdict.values())):
                self._check_data(tdict)
        except ValueError:
            return False
        else:
            return True

    @abc.abstractmethod
    def _check_attrs(self, attrs: AttrDict, /) -> None:
        """
        Raise `ValueError` if `self` cannot attach to tdict with `attrs`.
        """

    @abc.abstractmethod
    def _check_data(self, tdict: td.TensorDict, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or cannot attach to `tdict`.
        """
