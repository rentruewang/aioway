# Copyright (c) AIoWay Authors - All Rights Reserved

"""
Domains of tensors whose dimensions are tagged: lists, batches, temporal, spatial.

Lists check dimensions from the right: a list `ndim` deep checks dimension `-ndim`.
An image is `List[Spatial[Spatial]]`, `[C, H, W]`.
Batch is implicit: any leading dimensions in front of the ones a domain describes.
"""

import abc
import dataclasses as dcls
import typing

import torch

import aioway.t.attrs.shapes
from aioway.t.attrs import DTypeFamily, Shape, parse_attr

from .domains import AnyDomain, Domain, InvalidValueForDomain

__all__ = [
    "TensorDomain",
    "DTypeFamilyMixin",
    "RequiresGradMixin",
    "RangeMixin",
    "IntTensorDomain",
    "FloatTensorDomain",
    "ScoreTensorDomain",
    "SizedMixin",
    "ListDomain",
    "TemporalDomain",
    "SpatialDomain",
    "SizedListDomain",
    "SizedTemporalDomain",
    "SizedSpatialDomain",
    "grid_of",
    "image",
    "sized_image",
    "video",
    "sized_video",
]


# Tensors ====


class TensorDomain(Domain[torch.Tensor]):
    "Any tensor."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if not isinstance(obj, torch.Tensor):
            raise InvalidValueForDomain(f"Expected a tensor, got {type(obj)=}.")


# Tensor mixins ====


class DTypeFamilyMixin(TensorDomain):
    "A tensor of this dtype family."

    family: DTypeFamily = dcls.field(kw_only=True)
    "The dtype family the tensor must be in."

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        if (family := parse_attr(obj).dtype.family) != self.family:
            raise InvalidValueForDomain(
                f"dtype family {family}, expected {self.family}"
            )


class RequiresGradMixin(TensorDomain):
    "A tensor that requires grad or not."

    requires_grad: bool = dcls.field(kw_only=True)
    "Whether the tensor must require grad."

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        if (req_grad := parse_attr(obj).requires_grad) != self.requires_grad:
            raise InvalidValueForDomain(f"{req_grad=}, expected {self.requires_grad}")


class RangeMixin(TensorDomain):
    "A tensor whose values are in `[low, high]`."

    low: float = dcls.field(kw_only=True)
    "The smallest value allowed, inclusive."

    high: float = dcls.field(kw_only=True)
    "The largest value allowed, inclusive."

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)

        if not ((obj >= self.low) & (obj <= self.high)).all():
            raise InvalidValueForDomain(f"values not in [{self.low}, {self.high}]")


# Dtypes ====


class IntTensorDomain(DTypeFamilyMixin):
    "An int tensor."

    family: DTypeFamily = dcls.field(default="int", init=False)
    "Always int."

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]


class FloatTensorDomain(DTypeFamilyMixin):
    "A float tensor."

    family: DTypeFamily = dcls.field(default="float", init=False)
    "Always float."

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]


class ScoreTensorDomain(FloatTensorDomain):
    "A float tensor of shape `()`, e.g. a reduced loss."

    @property
    def __extends__(self) -> list[Domain]:
        return [FloatTensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        if shape := parse_attr(obj).shape:
            raise InvalidValueForDomain(f"a score has shape (), got {shape=}")


# Lists ====


class ListDomain(TensorDomain):
    "`inner` values in order, `[L, ...]`."

    inner: Domain = AnyDomain()

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

    @property
    def ndim(self) -> int:
        "How many trailing dimensions this list and the lists inside take."

        return 1 + (self.inner.ndim if isinstance(self.inner, ListDomain) else 0)

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        shape = parse_attr(obj).shape
        if len(shape) < self.ndim:
            raise InvalidValueForDomain(f"needs {self.ndim} dims, got {shape=}")

        # Removing this list's dimension keeps the batch in front for `inner`.
        if shape[-self.ndim]:
            self.inner.check_type(obj.select(-self.ndim, 0))

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)

        # Data checks are on values, which do not change when a dimension is peeled.
        self.inner.check_data(obj)


# Mixins ====


class SizedMixin(ListDomain, abc.ABC):
    """
    A list of this length, providing `shape`, built from the `inner`s.

    Batch is not part of `shape`, `-1` means any length.
    """

    length: int = dcls.field(kw_only=True)

    @property
    def shape(self) -> Shape:
        # `inner` is a field of the list this is mixed into.
        inner = self.inner
        rest: aioway.t.attrs.shapes.Shape | list[int] = []

        if isinstance(inner, SizedMixin):
            rest = inner.shape
        elif isinstance(inner, ListDomain):
            rest = [-1] * inner.ndim

        return Shape.parse([self.length, *rest])

    def batch_size_of(self, obj: torch.Tensor, /) -> Shape:
        "The leading dimensions of `obj`, in front of the ones `shape` describes."

        shape = parse_attr(obj).shape
        if len(shape) < self.ndim:
            raise InvalidValueForDomain(f"needs {self.ndim} dims, got {shape=}")

        return Shape.parse(list(shape)[: len(shape) - self.ndim])

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        length = parse_attr(obj).shape[-self.ndim]
        if self.length != -1 and length != self.length:
            raise InvalidValueForDomain(f"length {length}, expected {self.length}")


# Implementations ====


class TemporalDomain(ListDomain):
    "A list in time, `[T, ...]`."

    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


class SpatialDomain(ListDomain):
    "A list in space, `[S, ...]`."

    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


# Sized lists ====


class SizedListDomain(SizedMixin, ListDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


class SizedTemporalDomain(SizedListDomain, TemporalDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [
            TemporalDomain(self.inner),
            SizedListDomain(self.inner, length=self.length),
        ]


class SizedSpatialDomain(SizedListDomain, SpatialDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [
            SpatialDomain(self.inner),
            SizedListDomain(self.inner, length=self.length),
        ]


# Aliases ====


def grid_of(inner: Domain = AnyDomain()) -> SpatialDomain:
    "`Spatial[Spatial[inner]]`, `[H, W, ...]`."

    return SpatialDomain(SpatialDomain(inner))


def image() -> ListDomain:
    "`List[Spatial[Spatial]]`, `[C, H, W]`."

    return ListDomain(grid_of())


def sized_image(channels: int, height: int, width: int) -> SizedListDomain:
    "An image of this many channels, height and width."

    grid = SizedSpatialDomain(SizedSpatialDomain(length=width), length=height)
    return SizedListDomain(grid, length=channels)


def video() -> TemporalDomain:
    "`Temporal[List[Spatial[Spatial]]]`, `[T, C, H, W]`."

    return TemporalDomain(image())


def sized_video(
    frames: int, channels: int, height: int, width: int
) -> SizedTemporalDomain:
    "A video of this many frames, channels, height and width."

    return SizedTemporalDomain(sized_image(channels, height, width), length=frames)
