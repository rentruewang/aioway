# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of static attributes (`Attr`) and dynamic ones (`Tag`)."

import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from .attrs import Attr, Device, DType, Layout, Shape
from .tags import Tag, attach_tags, extract_tags

__all__ = ["Schema", "SchemaDict"]


@dcls.dataclass(frozen=True, slots=True)
class Schema:
    """
    Schema contains the full information of a tensor consumed by other `aioway` components.
    """

    attr: Attr
    "The attribute of a `torch.Tensor`."

    tags: dict[str, Tag] = dcls.field(default_factory=dict)
    "The tags attached to the `torch.Tensor`."

    def __post_init__(self):
        for tag in self.tags.values():
            tag.check_attr(self.attr)

    @property
    def shape(self) -> Shape:
        return self.attr.shape

    @property
    def dtype(self) -> DType:
        return self.attr.dtype

    @property
    def device(self) -> Device:
        return self.attr.device

    @property
    def layout(self) -> Layout:
        return self.attr.layout

    @property
    def requires_grad(self) -> bool:
        return self.attr.requires_grad

    def to_fake_tensor(self) -> torch.Tensor:
        """
        Convert the `Schema` to a `FakeTensor`, adding the tags.
        """

        tensor = self.attr.to_fake_tensor()
        attach_tags(tensor, *self.tags.values())
        return tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        """
        Create the schema from a `torch.Tensor`.
        """

        attr = Attr.parse(tensor)
        tags = extract_tags(tensor)
        return cls(attr, tags)


class SchemaDict(collections.UserDict[str, Schema]):
    """
    The dictionary of schemas.
    """

    @classmethod
    def from_tensor_mapping(
        cls, mapping: cabc.Mapping[str, torch.Tensor]
    ) -> typing.Self:
        """
        Create the schema dict from a mapping of `torch.Tensor`.
        """

        return cls({key: Schema.from_tensor(tensor) for key, tensor in mapping.items()})
