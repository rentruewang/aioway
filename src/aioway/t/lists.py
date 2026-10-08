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
    from aioway.t import Attr, Device, DType, Layout, Shape

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
        return hash(tuple(self._indexed.keys()))

    def __contains__(self, item: object) -> int:
        if isinstance(item, int | torch.Tensor):
            tensor_id = _get_id(item)
            return tensor_id in self._indexed

        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, TList):
            return list(self._indexed.keys()) == list(other._indexed.keys())

        if is_set_of(torch.Tensor):
            return sorted(self._indexed.keys()) == sorted(id(t) for t in other)

        if is_seq_of(torch.Tensor):
            return list(self._indexed.keys()) == [id(t) for t in other]

        return NotImplemented

    def __len__(self) -> int:
        return len(self._indexed)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensors[idx]

    def __iter__(self) -> cabc.Iterator[torch.Tensor]:
        return iter(self._tensors)

    def __add__(self, other: typing.Self) -> typing.Self:
        return self.from_indexed_tensors(self._indexed | other._indexed)

    def __sub__(self, other: typing.Self) -> typing.Self:
        return self.from_indexed_tensors(
            collections.OrderedDict(
                (k, self._indexed[k]) for k in self.keys() if k not in other.keys()
            )
        )

    def keys(self) -> cabc.KeysView[int]:
        "Get the keys for unordered comparison."

        return self._indexed.keys()

    def isdisjoint(self, other: TList) -> bool:
        return self.keys().isdisjoint(other.keys())

    def index(self, tensor: torch.Tensor | int) -> int:
        "Get the index of the tensor. O(n)."

        tensor = _get_id(tensor)

        if tensor not in self:
            raise ValueError("Tensor not in list.")

        for idx, t in enumerate(self._tensors):
            if tensor == id(t):
                return idx

        raise RuntimeError("Unreachable.")

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

    def shapes(self) -> list[Shape]:
        return [attr.shape for attr in self.attrs()]

    def dtypes(self) -> list[DType]:
        return [attr.dtype for attr in self.attrs()]

    def devices(self) -> list[Device]:
        return [attr.device for attr in self.attrs()]

    def layouts(self) -> list[Layout]:
        return [attr.layout for attr in self.attrs()]

    def requires_grads(self) -> list[bool]:
        return [attr.requires_grad for attr in self.attrs()]

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

    def flatten_tree(self: TList) -> tuple:
        return self._tensors, None

    def unflatten_tree(self: cabc.Iterable[torch.Tensor], _: None):
        return TList.from_iterable(self)

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


pyt.register_pytree_node(TList, TList.flatten_tree, TList.unflatten_tree)
