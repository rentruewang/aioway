# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of metadata describing the 'type' of data."

import dataclasses as dcls
import typing

import torch
from torch import nn

from aioway._torch import Attr, Device, DType, Layout, Shape

from .casts import CastedSpaceModule, register_cast
from .spaces import TensorSpace, space_dcls

__all__ = ["AttrSpace", "ShapeSpace"]


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

        schema: typing.Any = {}

        def add_if_not_none(key, val):
            if val is None:
                return

            schema[key] = val

        add_if_not_none("shape", self.shape)
        add_if_not_none("dtype", self.dtype)
        add_if_not_none("device", self.device)
        add_if_not_none("layout", self.layout)
        add_if_not_none("requires_grad", self.requires_grad)

        return Attr.parse(schema)

    @classmethod
    def from_attr(cls, attr: Attr, /) -> typing.Self:
        return cls(
            shape=attr.shape,
            device=attr.device,
            dtype=attr.dtype,
            layout=attr.layout,
            requires_grad=attr.requires_grad,
        )


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


@register_cast(ShapeSpace, ShapeSpace)
def flatten_tensor(input_space: ShapeSpace) -> CastedSpaceModule:
    flattened = input_space.shape.numel()
    output_space = ShapeSpace(Shape.parse(flattened))
    return CastedSpaceModule(output_space, module=nn.Flatten())
