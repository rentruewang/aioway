# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import enum
import functools
import re
import typing

import torch

from aioway.attrs import Attr

from .spaces import TensorSpace, space_dcls

__all__ = ["DimSpace", "DimInfo"]


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


@space_dcls
class DimSpace(TensorSpace):

    tags: str
    """
    The tags. For efficiency purposes we store it in strings.
    Strings are more compact and we can use regex.
    """

    def __post_init__(self) -> None:
        if not _valid_dim_tag(self.tags):
            raise ValueError("The dimension tags are not valid.")

    @typing.override
    def _check_attr(self, attr: Attr) -> None:
        if len(self.tags) != (ndim := attr.ndim):
            raise ValueError(
                f"The {self.tags=} dimensions do not match the tensor's {ndim=}."
            )

    @typing.override
    def _check_data(self, data: torch.Tensor) -> None:
        if DimInfo.PROB.value not in self.tags:
            return

        # Probability dims should sum to 1.
        prob_dims = [i for i, tag in enumerate(self.tags) if tag == DimInfo.PROB.value]

        for dim in prob_dims:
            summation = data.sum(dim=dim)
            ones = torch.ones_like(summation)

            if not torch.allclose(summation, ones):
                raise ValueError

    def _sample_n(self, batch_size: int):
        tags = [batch_size if t == "b" else 1 for t in self.tags]
        return torch.randn(*tags)


def _valid_dim_tag(tags: str) -> bool:
    return _valid_regex().fullmatch(tags) is not None


@functools.cache
def _valid_regex():
    values = [d.value for d in DimInfo]
    group = "(" + "|".join(values) + ")"
    return re.compile(f"{group}*")
