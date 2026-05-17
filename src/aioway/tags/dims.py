# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import dataclasses as dcls
import enum
import functools
import re

from .tags import Tag

__all__ = ["DimTag", "DimInfo", "valid_dim_tag"]


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


@dcls.dataclass(frozen=True, slots=True)
class DimTag(Tag):
    tags: str
    """
    The tags. For efficiency purposes we store it in strings.
    Strings are more compact and we can use regex.
    """

    def __post_init__(self) -> None:
        if len(self.tags) != (ndim := self.ndim):
            raise ValueError(
                f"The {self.tags=} dimensions do not match the tensor's {ndim=}."
            )

        if not valid_dim_tag(self.tags):
            raise ValueError("The dimension tags are not valid.")


def valid_dim_tag(tags: str) -> bool:
    return _valid_regex().fullmatch(tags) is not None


@functools.cache
def _valid_regex():
    values = [d.value for d in DimInfo]
    group = "(" + "|".join(values) + ")"
    return re.compile(f"{group}*")
