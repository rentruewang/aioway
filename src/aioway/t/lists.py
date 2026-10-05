# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

from aioway._utils import is_frozenset_of
from aioway._utils import is_set_of
import collections
import functools
import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pytree

from aioway._utils import is_seq_of

from .attrs import parse_attr
from .overrides import is_fake

if typing.TYPE_CHECKING:
    from .attrs import Attr

__all__ = ["TList"]


@typing.final
class TList:
    """
    This is an immutable list of tensors, supporting fast lookups.

    It deduplicates the tensors it received.

    The list is stable (insertion order = getitem order).
    """

    def __init__(self, tensors: cabc.Iterable[torch.Tensor]) -> None:
        self._indexed = collections.OrderedDict((id(t), t) for t in tensors)
        self._tensors = tuple(self._indexed.values())

    def __repr__(self) -> str:
        body = ",".join(map(str, (parse_attr(t) for t in self)))
        return f"[{body}]"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._indexed.keys())))

    def __contains__(self, item: object) -> int:
        if isinstance(item, int | torch.Tensor):
            tensor_id = _get_id(item)
            return tensor_id in self._indexed

        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, TList):
            return self._indexed.keys() == other._indexed.keys()

        if is_set_of(torch.Tensor) or is_frozenset_of(torch.Tensor):
            return sorted(self._indexed.keys()) == sorted(id(t) for t in other)

        if is_seq_of(torch.Tensor):
            return list(self._indexed.keys()) == [id(t) for t in other]

        return NotImplemented

    def __len__(self) -> int:
        return len(self._indexed)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensors[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index(self, tensor: torch.Tensor | int) -> int:
        "Get the index of the tensor. O(1)."

        tensor = _get_id(tensor)

        if tensor not in self:
            raise ValueError("Tensor not in list.")

        for idx, t in enumerate(self._tensors):
            if tensor == id(t):
                return idx

        raise IndexError

    @functools.cached_property
    def any_fake(self) -> bool:
        "Check if this contains any fake items."
        return any(is_fake(t) for t in self)

    @functools.cached_property
    def all_fake(self) -> bool:
        "Check if all items are fake."
        return all(is_fake(t) for t in self)

    def attrs(self) -> list[Attr]:
        """
        Convert `TList` to a list of `Attr`.
        """

        return parse_attr(self)


# Utility functions. ====


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    raise TypeError(type(tensor))


# Register for pytree. ====


def _flatten_tlist(tlist: TList):
    return list(tlist), None


def _unflatten_tlist(tlist: cabc.Iterable[torch.Tensor], _: None):
    return TList(tlist)


pytree.register_pytree_node(TList, _flatten_tlist, _unflatten_tlist)
