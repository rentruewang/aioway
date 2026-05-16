# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import dataclasses as dcls
import enum
import functools
import re
import typing

import torch

from aioway.fake import is_fake_tensor

__all__ = ["set_dim_tag", "get_dim_tag", "check_dim_tag", "DimTag"]


@typing.runtime_checkable
class HasDimTag(typing.Protocol):
    __aioway_dim_tag__: DimTag


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


@dcls.dataclass(frozen=True)
class DimTag:
    tensor: torch.Tensor
    "The tensor that is being piggy backed."

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

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def is_fake(self) -> bool:
        return is_fake_tensor(self.tensor)


def set_dim_tag(tensor: torch.Tensor, tag: str):
    "Tag the `torch.Tensor` with the given tag."

    def _cast_tensor(t: typing.Any) -> HasDimTag:
        return t

    dim_tag = DimTag(tensor, tag)
    tagged = _cast_tensor(tensor)
    tagged.__aioway_dim_tag__ = dim_tag


def get_dim_tag(tensor: torch.Tensor) -> DimTag | None:
    if isinstance(tensor, HasDimTag):
        return tensor.__aioway_dim_tag__

    else:
        return None


def check_dim_tag(tags: str) -> bool:
    return _valid_regex().fullmatch(tags) is not None


@functools.cache
def _valid_regex():
    values = [d.value for d in DimInfo]
    group = "(" + "|".join(values) + ")"
    return re.compile(f"{group}*")
