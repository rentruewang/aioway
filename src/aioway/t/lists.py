# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

import collections
import functools
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import is_seq_of

from .overrides import is_fake

__all__ = ["TList"]


@typing.final
class TList:
    """
    This is an immutable list of tensors, supporting fast lookups.

    It deduplicates the tensors it received.
    """

    def __init__(self, tensors: cabc.Iterable[torch.Tensor]) -> None:
        tensor_with_ids = sorted(((id(t), t) for t in tensors), key=lambda it: it[0])
        self.__indexed = collections.OrderedDict(tensor_with_ids)

        # Also store the ids s.t. we have fast `__getitem__`.
        self.__ids = tuple(self.__indexed)

        assert len(self.__indexed) == len(self.__ids)

    def __hash__(self) -> int:
        return hash(self.__ids)

    def __contains__(self, item: object) -> int:
        if isinstance(item, int | torch.Tensor):
            tensor_id = _get_id(item)
            return tensor_id in self.__indexed

        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, TList):
            return self.__indexed.keys() == other.__indexed.keys()

        if is_seq_of(torch.Tensor):
            return list(self.__ids) == sorted(id(t) for t in other)

        return NotImplemented

    def __len__(self) -> int:
        return len(self.__indexed)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.__indexed[self.__ids[idx]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index(self, tensor: torch.Tensor | int) -> int:
        "Get the index of the tensor."
        if tensor not in self:
            raise ValueError("Tensor not in list.")

        return int(np.searchsorted(self.__ids, _get_id(tensor)))

    @functools.cached_property
    def any_fake(self) -> bool:
        "Check if this contains any fake items."

        return any(is_fake(t) for t in self)

    @functools.cached_property
    def all_fake(self) -> bool:
        "Check if all items are fake."

        return any(is_fake(t) for t in self)


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    raise TypeError(type(tensor))
