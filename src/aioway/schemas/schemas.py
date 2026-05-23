# Copyright (c) AIoWay Authors - All Rights Reserved

"Schema is a collection of static attributes (`Attr`) and dynamic ones (`Tag`)."

import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from .attrs import Attr
from .tags import Tag, extract_tags

__all__ = ["Schema", "SchemaDict"]


@dcls.dataclass(frozen=True, slots=True)
class Schema:
    """
    Schema contains the full information of a tensor consumed by other `aioway` components.
    """

    attr: Attr
    "The attribute of a `torch.Tensor`."

    tags: dict[str, Tag]
    "The tags attached to the `torch.Tensor`."

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
