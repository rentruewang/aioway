# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.attrs import Attr, Device, DType, Layout, Shape

from .tags import TensorTag, tags_dcls

__all__ = ["AttrTag"]


@tags_dcls
class AttrTag(TensorTag):
    """
    Tag that specify `Attr`s that should be respected.
    """

    NAME = "__aioway_attr__"

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
