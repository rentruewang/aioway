# Copyright (c) AIoWay Authors - All Rights Reserved

"A utility for a list of tensor."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway._utils import is_seq_of, is_set_of
from aioway.t.attrs import parse_attr
from aioway.t.overrides import is_fake_tensor
from aioway.t.pytrees import (
    KeyEntry,
    PyTreeFlatten,
    PyTreeFlattenWithKey,
    PyTreeMixin,
    SequenceKey,
)

if typing.TYPE_CHECKING:
    from aioway.t import Attr, Device, DType, Layout, Shape

__all__ = ["TList"]


@typing.final
class TList(PyTreeMixin, cabc.Sequence[torch.Tensor]):
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

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, torch.Tensor):
            return False

        for tensor in self._tensors:
            if item is tensor:
                return True

        return False

    def __eq__(self, other) -> bool:
        def ids(tensors: cabc.Sequence[torch.Tensor]) -> list[int]:
            return [id(t) for t in tensors]

        # If the RHS is a `list` or `TList`, care about ordering.

        if isinstance(other, TList):
            return ids(self._tensors) == ids(other._tensors)

        if is_seq_of(torch.Tensor)(other):
            return ids(self._tensors) == ids(other)

        # If the RHS is a `set`, don't care about ordering.

        if is_set_of(torch.Tensor)(other):
            return frozenset(self._tensors) == frozenset(other)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._tensors)

    @typing.overload
    def __getitem__(self, i: int) -> torch.Tensor: ...

    @typing.overload
    def __getitem__(self, i: slice) -> typing.Self: ...

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._tensors[i]

        if isinstance(i, slice):
            return type(self)(self._tensors[i])

        raise IndexError(type(i))

    def __iter__(self) -> cabc.Iterator[torch.Tensor]:
        return iter(self._tensors)

    def __add__(self, other: typing.Self) -> typing.Self:
        return type(self)(self._tensors + other._tensors)

    def __sub__(self, other: typing.Self) -> typing.Self:
        return type(self)([tensor for tensor in self if tensor not in other])

    def __or__(self, other: typing.Self) -> typing.Self:
        return self.deduplicate(self + other)

    def isdisjoint(self, other: TList) -> bool:
        return set(self).isdisjoint(other)

    def index(
        self, tensor: torch.Tensor, start: int = 0, stop: int | None = None, /
    ) -> int:
        "Get the index of the tensor. O(n)."

        for i in range(len(self))[start:stop]:
            elem = self[i]

            if tensor is elem:
                return i

        raise ValueError("TList.index(tensor): tensor not in list.")

    @functools.cached_property
    def fakeness(self) -> FakeNess:
        "Return the fake ness of the tensors."
        return FakeNess.from_tensors(self)

    @property
    def has_fake(self) -> bool:
        return self.fakeness.has_fake

    @property
    def all_fake(self) -> bool:
        return self.fakeness.all_fake

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

    @property
    def ids(self) -> list[int]:
        return [id(t) for t in self]

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
        return TList(tuple(children))


def _seq_key(idx: int) -> KeyEntry:
    return typing.cast(typing.Any, SequenceKey)(idx)


# Helper objects ====


@dcls.dataclass(frozen=True)
class FakeNess:
    "The fakeness of a list of tensors."

    all_fake: bool
    "All of them is fake."

    has_fake: bool
    "Some of them is fake."

    @property
    def all_real(self) -> bool:
        return not self.has_fake

    @property
    def has_real(self) -> bool:
        return not self.all_fake

    @classmethod
    def from_tensors(cls, tensors: cabc.Iterable[torch.Tensor], /) -> typing.Self:
        fakes = [is_fake_tensor(t) for t in tensors]
        return cls(all_fake=all(fakes), has_fake=any(fakes))
