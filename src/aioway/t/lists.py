# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

import collections
import functools
from collections import abc as cabc

import torch

from .overrides import is_fake

__all__ = ["TList"]


class TList:
    """
    This is an immutable list of tensors, supporting fast lookups.

    It deduplicates the tensors it received.
    """

    def __init__(self, tensors: cabc.Iterable[torch.Tensor]) -> None:
        self.__indexed = collections.OrderedDict((id(t), t) for t in tensors)

        # Also store the tensors themselves s.t. we have fast `__getitem__`.
        self.__tensors = tuple(self.__indexed.values())

    def __hash__(self) -> int:
        return hash(tuple(self.__indexed))

    def __contains__(self, item: object) -> int:
        if isinstance(item, int | torch.Tensor):
            tensor_id = _get_id(item)
            return tensor_id in self.__indexed

        return False

    def __len__(self) -> int:
        return len(self.__indexed)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.__tensors[idx]

    def __iter__(self) -> cabc.Iterator[torch.Tensor]:
        return iter(self.__tensors)

    @functools.cached_property
    def any_fake(self) -> bool:
        "Check if this contains any fake items."

        return any(is_fake(t) for t in self.__tensors)

    @functools.cached_property
    def all_fake(self) -> bool:
        "Check if all items are fake."

        return any(is_fake(t) for t in self.__tensors)


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    raise TypeError(type(tensor))
