# Copyright (c) AIoWay Authors - All Rights Reserved

"""
Domains of tensors whose dimensions are tagged: lists, batches, temporal, spatial.

Each list peels the first dimension and checks `inner` on the rest.
An image is `Batch[List[Spatial[Spatial]]]`, `[B, C, H, W]`.
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
    "SizedMixin",
    "ListDomain",
    "BatchDomain",
    "TemporalDomain",
    "SpatialDomain",
    "SizedListDomain",
    "SizedBatchDomain",
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


# Lists ====


class ListDomain(TensorDomain):
    "`inner` values in order, `[L, ...]`."

    inner: Domain = AnyDomain()

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        shape = parse_attr(obj).shape
        if not len(shape):
            raise InvalidValueForDomain(f"a list is [L, ...], got {shape=}")

        if shape[0]:
            self.inner.check_type(obj.select(0, 0))

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)

        # Data checks are on values, which do not change when a dimension is peeled.
        self.inner.check_data(obj)


# Mixins ====


class SizedMixin(ListDomain, abc.ABC):
    """
    A list of this length, providing `shape`, built from the sized `inner`s.

    Batch is part of `shape`, `length=-1` means any length.
    """

    length: int = dcls.field(kw_only=True)

    @property
    def shape(self) -> Shape:
        # `inner` is a field of the list this is mixed into.
        inner = self.inner
        rest: aioway.t.attrs.shapes.Shape | list[int] = (
            inner.shape if isinstance(inner, SizedMixin) else []
        )
        return Shape.parse([self.length, *rest])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        length = parse_attr(obj).shape[0]
        if self.length != -1 and length != self.length:
            raise InvalidValueForDomain(f"length {length}, expected {self.length}")


# Implementations ====


class BatchDomain(ListDomain):
    "A list of examples, `[B, ...]`."

    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


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


class SizedBatchDomain(SizedListDomain, BatchDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [
            BatchDomain(self.inner),
            SizedListDomain(self.inner, length=self.length),
        ]


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


def image() -> BatchDomain:
    "`Batch[List[Spatial[Spatial]]]`, `[B, C, H, W]`."

    return BatchDomain(ListDomain(grid_of()))


def sized_image(channels: int, height: int, width: int) -> SizedBatchDomain:
    "An image of this many channels, height and width, any batch size."

    grid = SizedSpatialDomain(SizedSpatialDomain(length=width), length=height)
    return SizedBatchDomain(SizedListDomain(grid, length=channels), length=-1)


def video() -> BatchDomain:
    "`Batch[Temporal[List[Spatial[Spatial]]]]`, `[B, T, C, H, W]`."

    return BatchDomain(TemporalDomain(ListDomain(grid_of())))


def sized_video(
    frames: int, channels: int, height: int, width: int
) -> SizedBatchDomain:
    "A video of this many frames, channels, height and width, any batch size."

    img = sized_image(channels, height, width)
    return SizedBatchDomain(SizedTemporalDomain(img.inner, length=frames), length=-1)
