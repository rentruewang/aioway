# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import enum
import functools
import re
import typing

from ..attrs import Attr
from .tags import Tag, tags_dcls

__all__ = ["DimTag", "DimInfo"]


class DimInfo(enum.StrEnum):
    SPATIAL = "s"
    """
    Marks a dimension as contiuous in space. This can be used on CNN etc.
    """

    TEMPORAL = "t"
    """
    Marks a dimension as contiuous in time. This can be used on transformers / RNN etc.
    """

    INDEPENDENT = "i"
    """
    Marks a dimension as the batch dimension.
    All elements across this dimension should be independent.
    """

    PROB = "p"
    """
    Mark a dimension as a probability dimension,
    which means that the items in this dimension should sum to a constant.

    Only applies to positive tensors.
    """

    OTHER = "x"
    """
    An x means we have no info in this dimension.
    """


@tags_dcls
class DimTag(Tag):
    NAME = "__aioway_dim_tag__"

    tags: str
    """
    The tags. For efficiency purposes we store it in strings.
    Strings are more compact and we can use regex.
    """

    @typing.override
    def check_attr(self, attr: Attr) -> None:
        if len(self.tags) != (ndim := attr.ndim):
            raise ValueError(
                f"The {self.tags=} dimensions do not match the tensor's {ndim=}."
            )

    @typing.override
    def check_self(self) -> None:
        if not _valid_dim_tag(self.tags):
            raise ValueError("The dimension tags are not valid.")


def _valid_dim_tag(tags: str) -> bool:
    return _valid_regex().fullmatch(tags) is not None


@functools.cache
def _valid_regex():
    values = [d.value for d in DimInfo]
    group = "(" + "|".join(values) + ")"
    return re.compile(f"{group}*")
