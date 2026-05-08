# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import dataclasses as dcls

__all__ = ["Info", "IsImage", "IsVideo"]


@dcls.dataclass(frozen=True)
class Info:
    """
    The extra kinds of information that is stored in the attributes.
    """


class IsImage(Info):
    """
    Mark a tensor as image.

    This means that the floating tensor values should all be between 0 to 1.
    """

    ordering: tuple[str, ...] = "c", "w", "h"
    """
    The channel ordering.
    """


class IsVideo(Info):
    """
    Mark a tensor as video.

    This means that the floating tensor values should all be between 0 to 1.
    """

    ordering: tuple[str, ...] = "t", "c", "w", "h"
    """
    The channel ordering.
    """


@dcls.dataclass(frozen=True)
class _SingleDimMixin:
    dim: int
    """
    The dimension marked.
    """

    def __int__(self) -> int:
        return self.dim


@dcls.dataclass(frozen=True)
class ProbDim(_SingleDimMixin, Info):
    """
    Marks a dimension as probablity dimension (sums to 1).
    The dimension should have all >= 0 elements as well.

    Note: One hot also satisfy this criteria.
    """


@dcls.dataclass(frozen=True)
class TimeDim(_SingleDimMixin, Info):
    """
    Marks a dimension as time contiuous. This can be used on transformers / RNN etc.
    """


@dcls.dataclass(frozen=True)
class SpaceDim(_SingleDimMixin, Info):
    """
    Marks a dimension as space contiuous. This can be used on transformers / RNN etc.
    """
