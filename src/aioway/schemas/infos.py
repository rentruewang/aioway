# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import collections
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np

from aioway._common import IntArray

__all__ = [
    "Info",
    "InfoList",
    "IsImage",
    "IsVideo",
    "BatchDim",
    "ProbDim",
    "TimeDim",
    "SpaceDim",
]


@dcls.dataclass(frozen=True)
class Info:
    """
    The extra kinds of information that is stored in the attributes.
    """


class InfoList:
    """
    An info list that is both a list and a dict (organized by list).
    """

    def __init__(self, *infos: Info):
        self._info_by_type: dict[type[Info], list[Info]] = _categorize_infos(infos)
        "Organize info by their types."

        if not all(issubclass(info, Info) for info in self._info_by_type):
            raise TypeError(f"Not all info in {self._info_by_type=} is a `Info`.")

        if not all(
            isinstance(info_inst, info_type)
            for (info_type, info_list) in self._info_by_type.items()
            for info_inst in info_list
        ):
            raise TypeError(
                "Invalid info passed in. Infos should be a mapping of type to list of instances. "
                f"But got {self._info_by_type}."
            )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InfoList):
            return self._info_by_type == other._info_by_type

        if isinstance(other, dict):
            return self._info_by_type == other

        if isinstance(other, list):
            return list(self) == other

        return NotImplemented

    def __repr__(self):
        return ",".join(map(str, self))

    def __getstate__(self):
        return {
            key.__qualname__: [info.__getstate__() for info in val]
            for key, val in self._info_by_type.items()
        }

    def __len__(self):
        return self._info_cumsum_items[-1]

    @typing.overload
    def __getitem__(self, info: type[Info]) -> list[Info]: ...

    @typing.overload
    def __getitem__(self, info: int) -> Info: ...

    def __getitem__(self, info):
        if isinstance(info, int):
            # Mute the typing errors and shorten it s.t. the lines don't wrap.
            search: typing.Any = np.searchsorted
            cumsum = self._info_cumsum_items

            idx: int = search(cumsum, side="right")

            if idx > len(cumsum):
                raise IndexError(f"Index out of bounds, {len(self)=}.")

            return self._info_by_type[self._key_list[idx]][idx - cumsum[idx]]

        if isinstance(info, type) and issubclass(info, Info):
            return self._info_by_type[info]

        raise TypeError(f"Can not handle {info=}.")

    def __iter__(self):
        for info_list in self.values():
            yield from info_list

    @functools.cached_property
    def _info_cumsum_items(self) -> IntArray:
        "The cumsum for the info."
        val_lens = [len(val) for val in self.values()]
        return np.cumsum(val_lens).tolist()

    @functools.cached_property
    def _key_list(self) -> list[type[Info]]:
        "The cumsum for the info."
        return list(self.keys())

    def keys(self):
        return self._info_by_type.keys()

    def values(self):
        return self._info_by_type.values()

    def items(self):
        return self._info_by_type.items()

    def by_type(self):
        return self._info_by_type


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
class BatchDim(_SingleDimMixin, Info):
    """
    Marks a dimension as the batch dimension.

    This dimension is assumed to be decomposbile.
    """


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
    Marks a dimension as contiuous in time. This can be used on transformers / RNN etc.
    """


@dcls.dataclass(frozen=True)
class SpaceDim(_SingleDimMixin, Info):
    """
    Marks a dimension as contiuous in space. This can be used on CNN etc.
    """


def _categorize_infos(
    infos: cabc.Iterable[Info],
) -> dict[type[Info], list[Info]]:
    """
    Organize sequence of `Info` by `type(info)`, into a `dict[type, list]`.
    """

    result: dict[type[Info], list[Info]] = collections.defaultdict(list)

    for info in infos:
        result[type(info)].append(info)

    return result
