# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

import functools
import typing
from collections import abc as cabc

import torch

from aioway._pytrees import (
    KeyEntry,
    PyTreeFlatten,
    PyTreeFlattenWithKey,
    PyTreeMixin,
    SequenceKey,
)
from aioway._utils import is_seq_of, is_set_of
from aioway.t.attrs import parse_attr
from aioway.t.overrides import has_fake

if typing.TYPE_CHECKING:
    from aioway.t import Attr, Device, DType, Layout, Shape

__all__ = ["TList"]


@typing.final
class TList(PyTreeMixin):
    """
    This is an immutable list of tensors, supporting fast lookups.
    It deduplicates the tensors it received.
    The list is stable (insertion order = getitem order).

    Note: this relies on `dict` values being ordered.

    Prefer classmethods over its constructor.

    When doing `==` comparison, if the RHS is a `Set` the order doesn't matter,
    but if it's a `Sequence` or `TList` the order does matter.
    """

    def __init__(self, tensors: cabc.Sequence[torch.Tensor] = (), /) -> None:
        self._tensors = tuple(tensors)

    def __repr__(self) -> str:
        body = ",".join(map(str, (parse_attr(t) for t in self)))
        return f"[{body}]"

    def __hash__(self) -> int:
        return hash(self._tensors)

    def __contains__(self, item: object) -> int:
        if not isinstance(item, torch.Tensor):
            return False

        for tensor in self._tensors:
            if item is tensor:
                return True

        return False

    def __eq__(self, other) -> bool:
        def ids(tensors: cabc.Sequence[torch.Tensor]) -> list[int]:
            return [id(t) for t in tensors]

        if isinstance(other, TList):
            return ids(self._tensors) == ids(other._tensors)

        if is_seq_of(torch.Tensor)(other):
            return ids(self._tensors) == ids(other)

        if is_set_of(torch.Tensor)(other):
            return frozenset(self._tensors) == frozenset(other)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensors[idx]

    def __iter__(self) -> cabc.Iterator[torch.Tensor]:
        return iter(self._tensors)

    def __sub__(self, other: typing.Self) -> typing.Self:
        return type(self)([tensor for tensor in self if tensor not in other])

    def __or__(self, other: typing.Self) -> typing.Self:
        joint = [*self, *other]
        return self.deduplicate(joint)

    def isdisjoint(self, other: TList) -> bool:
        return set(self).isdisjoint(other)

    def index(self, tensor: torch.Tensor) -> int:
        "Get the index of the tensor. O(n)."

        for idx, t in enumerate(self._tensors):
            if tensor is t:
                return idx

        raise ValueError("TList.index(tensor): tensor not in list.")

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
    def deduplicate(cls, tensors: cabc.Iterable[torch.Tensor], /) -> typing.Self:
        "Convert from an iterable of tensors."

        ids: set[int] = set()
        dedup: list[torch.Tensor] = []

        for tensor in tensors:
            if (tensor_id := id(tensor)) in ids:
                continue

            ids.add(tensor_id)
            dedup.append(tensor)

        return cls(dedup)

    def _tree_flatten(self) -> PyTreeFlatten:
        return list(self._tensors), None

    def _tree_flatten_with_keys(self) -> PyTreeFlattenWithKey:
        with_path = [(_seq_key(i), t) for i, t in enumerate(self._tensors)]
        return with_path, None

    @classmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        assert context is None
        return TList.deduplicate(children)


def _seq_key(idx: int) -> KeyEntry:
    return typing.cast(typing.Any, SequenceKey)(idx)
