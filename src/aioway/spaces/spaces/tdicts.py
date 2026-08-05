# Copyright (c) AIoWay Authors - All Rights Reserved

"A collection of tensors and their `Space`s."

import abc
import typing

import tensordict as td
import torch

from aioway._api import public_api
from aioway._torch import Schema, is_real_tensor

from .spaces import Space, space_dcls

__all__ = ["IsTdictSpace", "TdictSpace"]


@public_api
@space_dcls
class IsTdictSpace(Space[td.TensorDict]):
    "Constrains only that the item is a `td.TensorDict`."

    @typing.override
    @typing.final
    def contains(self, tdict: td.TensorDict, /) -> bool:
        return isinstance(tdict, td.TensorDict)

    def _sample_n(self, n):
        return td.TensorDict({"a": torch.zeros([n])})


@public_api
@space_dcls
class TdictSpace(Space[td.TensorDict]):
    "A `Space` that checks a `td.TensorDict`."

    @typing.override
    @typing.final
    def contains(self, tdict: td.TensorDict, /) -> bool:
        if not isinstance(tdict, td.TensorDict):
            raise TypeError(f"{type(tdict)} is not a `td.TensorDict`.")

        attrs = Schema.parse(tdict)

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
    def _check_attrs(self, attrs: Schema, /) -> None:
        """
        Raise `ValueError` if `self` is incompatible with tdict with `attrs`.
        """

    @abc.abstractmethod
    def _check_data(self, tdict: td.TensorDict, /) -> None:
        """
        Raise `ValueError` if `self` is not valid or is incompatible with `tdict`.
        """
