# Copyright (c) AIoWay Authors - All Rights Reserved

"The contract that forces the checked data to be exact."

import dataclasses as dcls
import random
import typing
from collections import abc as cabc

import torch

from aioway.attrs import Attr, Device, DType, Layout, Shape

from .spaces import TensorSpace, space_dcls

__all__ = ["AttrSpace", "ShapeSpace", "DTypeSpace"]


@space_dcls
class ShapeSpace(TensorSpace):
    """
    Check if the `torch.Tensor` has the same shape.
    """

    shape: Shape
    """
    The shape to check.
    """

    def __len__(self) -> int:
        return len(self.shape)

    def __getitem__(self, idx: int) -> int:
        return self.shape[idx]

    def _check_attr(self, attr: Attr) -> None:
        if self.shape == attr.shape:
            return

        raise ValueError

    def _check_data(self, tensor: torch.Tensor) -> None:
        pass

    def _sample_n(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, *self.shape)


@space_dcls
class DTypeSpace(TensorSpace):
    """
    Check if the `torch.Tensor` has one of the dtypes.
    """

    dtypes: cabc.Sequence[DType]
    """
    The shape to check.
    """

    def _check_attr(self, attr: Attr) -> None:
        if attr.dtype not in self.dtypes:
            raise ValueError

    def _check_data(self, tensor: torch.Tensor) -> None:
        pass

    def _sample_n(self, batch_size: int) -> torch.Tensor:
        dtype = random.choice(self.dtypes)
        return torch.randn(batch_size).to(dtype.torch())


@space_dcls
class AttrSpace(TensorSpace):
    """
    Tag that specify `Attr`s that should be respected.
    """

    _: dcls.KW_ONLY

    shape: Shape | None = None
    "The given shape, if given, that must match."

    dtype: DType | None = None
    "The given dtype, if given, that must match."

    device: Device | None = None
    "The given device, if given, that must match."

    layout: Layout | None = None
    "The given layout, if given, that must match."

    requires_grad: bool | None = None
    "The given requires_grad, if given, that must match."

    def _check_attr(self, attr: Attr) -> None:
        def check_if_not_none[T](left: T | None, right: T):
            if left is None:
                return

            if left != right:
                raise ValueError

        check_if_not_none(self.shape, attr.shape)
        check_if_not_none(self.dtype, attr.dtype)
        check_if_not_none(self.device, attr.device)
        check_if_not_none(self.layout, attr.layout)
        check_if_not_none(self.requires_grad, attr.requires_grad)

    def _sample_n(self, batch_size: int) -> torch.Tensor:
        attr = self.to_attr()
        attr = dcls.replace(attr, shape=Shape.parse(batch_size, *attr.shape))
        return attr.to_fake_tensor()

    def _check_data(self, tensor: torch.Tensor) -> None:
        pass

    def to_attr(self) -> Attr:
        "Convert to `Attr`. If not enough info, will raise `TypeError`."

        attr_dict: typing.Any = {}

        def add_if_not_none(key, val):
            if val is None:
                return

            attr_dict[key] = val

        add_if_not_none("shape", self.shape)
        add_if_not_none("dtype", self.dtype)
        add_if_not_none("device", self.device)
        add_if_not_none("layout", self.layout)
        add_if_not_none("requires_grad", self.requires_grad)

        return Attr.parse(attr_dict)

    @classmethod
    def from_attr(cls, attr: Attr, /) -> typing.Self:
        return cls(
            shape=attr.shape,
            device=attr.device,
            dtype=attr.dtype,
            layout=attr.layout,
            requires_grad=attr.requires_grad,
        )
