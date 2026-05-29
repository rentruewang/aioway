# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import tensordict as td
import torch

from aioway.attrs import Attr, Device, DType, Layout, Shape

from .tags import Tag, TensorTag, tags_dcls

__all__ = [
    "HasShapeTag",
    "HasDTypeTag",
    "HasDeviceTag",
    "HasLayoutTag",
    "HasRequiresGradTag",
    "tag_attr",
]


def tag_attr(attr, item: torch.Tensor):
    "Tag the given tensor with the attributes, to mark constraints."

    HasShapeTag(attr.shape).attach(item)
    HasDeviceTag(attr.device).attach(item)
    HasDTypeTag(attr.dtype).attach(item)
    HasLayoutTag(attr.layout).attach(item)
    HasRequiresGradTag(attr.requires_grad).attach(item)


@tags_dcls
class HasShapeTag(Tag):
    """
    The tensor must have the shape given by this tag.
    """

    NAME = "__aioway_has_shape__"

    shape: Shape
    "The given shape that must match."

    @typing.override
    def check(self, item: torch.Tensor | td.TensorDict, /) -> bool:
        return item.shape == self.shape


@tags_dcls
class HasDTypeTag(Tag):
    """
    The tensor must have the dtype given by this tag.
    """

    NAME = "__aioway_has_dtype__"

    dtype: DType
    "The given dtype that must match."

    @typing.override
    def check(self, item: torch.Tensor | td.TensorDict, /) -> bool:
        return item.dtype == self.dtype


@tags_dcls
class HasDeviceTag(Tag):
    """
    The tensor must have the device given by this tag.
    """

    NAME = "__aioway_has_device__"

    device: Device
    "The given device that must match."

    @typing.override
    def check(self, item: torch.Tensor | td.TensorDict, /) -> bool:
        return item.device == self.device


@tags_dcls
class HasLayoutTag(TensorTag):
    """
    The tensor must have the layout given by this tag.
    """

    NAME = "__aioway_has_layout__"

    layout: Layout
    "The given layout that must match."

    @typing.override
    def _check_attr(self, attr: Attr):
        if attr.layout != self.layout:
            raise ValueError


@tags_dcls
class HasRequiresGradTag(Tag[torch.Tensor | td.TensorDict]):
    """
    The tensor must have the requires_grad given by this tag.
    """

    NAME = "__aioway_has_requires_grad__"

    requires_grad: bool
    "The given `requires_grad` attribute that must match."

    @typing.override
    def check(self, item: torch.Tensor | td.TensorDict, /) -> bool:
        return item.requires_grad == self.requires_grad
