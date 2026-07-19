# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

import torch

from aioway._api import public_api
from aioway._ufuncs import IdentityUFunc
from aioway.attrs import Attr
from aioway.errors import re_raise_func
from aioway.spaces import CoercionOutput, register_coercsion

from .spaces import TensorSpace, space_dcls

__all__ = ["ImageSpace", "ByteImageSpace", "FloatImageSpace"]


@public_api
@space_dcls
class ImageSpace(TensorSpace):
    "Images backed by an `uint8` tensor."

    num_channels: int
    "The number of channels in an image."

    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        assert attr.ndim == 4
        assert attr.shape[1] == self.num_channels

    @abc.abstractmethod
    def _check_data(self, tensor: torch.Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_n(self, n: int) -> torch.Tensor:
        raise NotImplementedError


@public_api
@space_dcls
class ByteImageSpace(ImageSpace):
    "Images backed by an `uint8` tensor."

    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr):
        super()._check_attr(attr)
        assert attr.dtype == "uint8"

    def _check_data(self, tensor: torch.Tensor):
        pass

    def _sample_n(self, n: int) -> torch.Tensor:
        return torch.randint(0, 256, [n, 3, 224, 224], dtype=torch.uint8)


@public_api
@space_dcls
class FloatImageSpace(ImageSpace):
    "Images backed by a `float` tensor."

    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr):
        super()._check_attr(attr)
        assert attr.dtype.family == "float"

    def _check_data(self, tensor: torch.Tensor):
        assert torch.all(tensor >= 0)
        assert torch.all(tensor < 1)

    def _sample_n(self, n: int) -> torch.Tensor:
        return torch.rand([n, 3, 224, 224], dtype=torch.float32)


@register_coercsion
def byte_is_image(img: ByteImageSpace) -> CoercionOutput[ImageSpace]:
    return CoercionOutput(img, IdentityUFunc())


@register_coercsion
def float_is_image(img: FloatImageSpace) -> CoercionOutput[ImageSpace]:
    return CoercionOutput(img, IdentityUFunc())
