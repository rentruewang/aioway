# Copyright (c) AIoWay Authors - All Rights Reserved

"`Space` for constraining the data and sampling."

import abc

from aioway._api import public_api
from aioway._utils import torch_fake_mode

from ..spaces import Space, space_dcls

__all__ = ["DataSpace", "AnyDataSpace"]


@public_api
@space_dcls
class DataSpace[T](Space[T], abc.ABC):
    def sample(self, n: int = 1) -> T:
        with torch_fake_mode():
            return self._sample_n(n)

    @abc.abstractmethod
    def _sample_n(self, n: int, /) -> T:
        raise NotImplementedError


@public_api
@space_dcls
class AnyDataSpace(DataSpace, abc.ABC):
    def contains(self, obj):
        return True

    def _sample_n(self, n: int, /):
        return object()
