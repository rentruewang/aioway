# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

import collections
import copy
import functools
import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pyt

from aioway._utils import is_seq_of, is_set_of

from .attrs import parse_attr
from .overrides import has_fake

if typing.TYPE_CHECKING:
    from .attrs import Attr

__all__ = ["TList"]


@typing.final
class TList:
    """
    This is an immutable list of tensors, supporting fast lookups.
    It deduplicates the tensors it received.
    The list is stable (insertion order = getitem order).

    It has a private constructor.

    When doing `==` comparison, if the RHS is a `Set` the order doesn't matter,
    but if it's a `Sequence` or `TList` the order does matter.
    """

    def __init__(
        self,
        *,
        indexed: collections.OrderedDict[int, torch.Tensor],
        tensors: tuple[torch.Tensor, ...],
    ) -> None:

        self._indexed = indexed
        self._tensors = tensors

        if len(self._indexed) != len(self._tensors):
            raise ValueError(
                "The indexed tensor dict is not the same length as the tensors."
            )

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
            return sorted(self._indexed.keys()) == sorted(other._indexed.keys())

        if is_set_of(torch.Tensor):
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

    def __add__(self, other: typing.Self) -> typing.Self:
        return self.from_indexed_tensors(self._indexed | other._indexed)

    def keys(self) -> cabc.KeysView[int]:
        "Get the keys for unordered comparison."

        return self._indexed.keys()

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
        return any(has_fake(t) for t in self)

    @functools.cached_property
    def all_fake(self) -> bool:
        "Check if all items are fake."
        return all(has_fake(t) for t in self)

    def attrs(self) -> list[Attr]:
        """
        Convert `TList` to a list of `Attr`.
        """

        return parse_attr(self)

    @classmethod
    def from_self_or_iter(
        cls, tensors: cabc.Iterable[torch.Tensor] | typing.Self, /
    ) -> typing.Self:
        # If it's a `TList` do a shallow copy.
        if isinstance(tensors, TList):
            return copy.copy(tensors)

        else:
            return cls.from_iterable(tensors)

    @classmethod
    def from_iterable(cls, tensors: cabc.Iterable[torch.Tensor], /) -> typing.Self:
        "Convert from an iterable of tensors."
        indexed = collections.OrderedDict((id(t), t) for t in tensors)
        return cls.from_indexed_tensors(indexed=indexed)

    @classmethod
    def from_indexed_tensors(
        cls, indexed: collections.OrderedDict[int, torch.Tensor]
    ) -> typing.Self:
        tensors = tuple(indexed.values())
        return cls(indexed=indexed, tensors=tensors)

    @classmethod
    def empty(cls) -> typing.Self:
        return cls.from_indexed_tensors(collections.OrderedDict())


# Utility functions. ====


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    raise TypeError(type(tensor))


# Register for pyt. ====


def _flatten_tlist(tlist: TList):
    return list(tlist), None


def _unflatten_tlist(tlist: cabc.Iterable[torch.Tensor], _: None):
    return TList.from_iterable(tlist)


pyt.register_pytree_node(TList, _flatten_tlist, _unflatten_tlist)
