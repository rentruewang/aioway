# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from aioway.attrs import Attr, Device, DType, Layout, Shape

from .tags import TensorTag, tags_dcls

__all__ = [
    "HasShapeTag",
    "HasDTypeTag",
    "HasDeviceTag",
    "HasLayoutTag",
    "HasRequiresGradTag",
    "tag_attr",
]


def tag_attr(attr: Attr, tensor: torch.Tensor):
    "Tag the given tensor with the attributes, to mark constraints."

    HasShapeTag(attr.shape).attach(tensor)
    HasDeviceTag(attr.device).attach(tensor)
    HasDTypeTag(attr.dtype).attach(tensor)
    HasLayoutTag(attr.layout).attach(tensor)
    HasRequiresGradTag(attr.requires_grad).attach(tensor)


@tags_dcls
class HasShapeTag(TensorTag):
    """
    The tensor must have the shape given by this tag.
    """

    NAME = "__aioway_has_shape__"

    shape: Shape
    "The given shape that must match."

    @typing.override
    def _check_attr(self, attr: Attr, /) -> None:
        assert attr.shape == self.shape


@tags_dcls
class HasDTypeTag(TensorTag):
    """
    The tensor must have the dtype given by this tag.
    """

    NAME = "__aioway_has_dtype__"

    dtype: DType
    "The given dtype that must match."

    @typing.override
    def _check_attr(self, attr: Attr, /) -> None:
        assert attr.dtype == self.dtype


@tags_dcls
class HasDeviceTag(TensorTag):
    """
    The tensor must have the device given by this tag.
    """

    NAME = "__aioway_has_device__"

    device: Device
    "The given device that must match."

    @typing.override
    def _check_attr(self, attr: Attr, /) -> None:
        assert attr.device == self.device


@tags_dcls
class HasLayoutTag(TensorTag):
    """
    The tensor must have the layout given by this tag.
    """

    NAME = "__aioway_has_layout__"

    layout: Layout
    "The given layout that must match."

    @typing.override
    def _check_attr(self, attr: Attr, /) -> None:
        assert attr.layout == self.layout


@tags_dcls
class HasRequiresGradTag(TensorTag):
    """
    The tensor must have the requires_grad given by this tag.
    """

    NAME = "__aioway_has_requires_grad__"

    requires_grad: bool
    "The given `requires_grad` attribute that must match."

    @typing.override
    def _check_attr(self, attr: Attr, /) -> None:
        assert attr.requires_grad == self.requires_grad
