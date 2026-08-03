# Copyright (c) AIoWay Authors - All Rights Reserved

"The base class for all `torch.Tensor` related `Shape`s."

import abc
import typing

import torch

from aioway._api import public_api
from aioway._torch import is_fake_tensor

from .spaces import Space, space_dcls

if typing.TYPE_CHECKING:
    from .attrs import Attr

__all__ = ["TensorSpace"]


@public_api
@space_dcls
class TensorSpace(Space[torch.Tensor], abc.ABC):
    "A `Space` that enforces constraints on a `torch.Tensor`."

    @typing.override
    @typing.final
    def contains(self, tensor: torch.Tensor, /) -> bool:
        from .attrs import Attr

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
        Raise `ValueError` if `self` is incompatible with tensor with `attr`.
        """

    @abc.abstractmethod
    def _check_data(self, tensor: torch.Tensor, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or is incompatible with `tensor`.
        """
