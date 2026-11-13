# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of lists: lists, temporal, spatial, grids, images, videos."

import dataclasses as dcls

import torch

from .domains import AnyDomain, Domain, InvalidDomain

__all__ = [
    "TensorDomain",
    "ListDomain",
    "TemporalDomain",
    "SpatialDomain",
    "GridDomain",
    "SizedGridDomain",
    "ImageDomain",
    "SizedImageDomain",
    "VideoDomain",
    "SizedVideoDomain",
]


# Tensors ====


class TensorDomain(Domain[torch.Tensor]):
    "Any tensor."

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if not isinstance(obj, torch.Tensor):
            raise InvalidDomain(f"expected a tensor, got {type(obj).__name__}")


# Lists ====


class ListDomain(Domain):
    """
    `inner` values in order. Either a tensor `[B, L, ...]`,
    or a sequence of `L` tensors, each `[B, ...]` (not contiguous).
    """

    inner: Domain = AnyDomain()

    @property
    def __parents__(self) -> list[Domain]:
        return [ListDomain(p) for p in self.inner.__parents__] or [AnyDomain()]

    def check_type(self, obj, /) -> None:
        super().check_type(obj)
        for item in _items(obj):
            self.inner.check_type(item)

    def check_data(self, obj, /) -> None:
        super().check_data(obj)
        for item in _items(obj):
            self.inner.check_data(item)


class TemporalDomain(ListDomain):
    "A list in time, `[B, T, ...]`."

    @property
    def __parents__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


class SpatialDomain(ListDomain):
    "A list in space, `[B, S, ...]`."

    @property
    def __parents__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


# Grids ====


class GridDomain(Domain):
    "Spatial[Spatial[inner]], `[B, H, W, ...]`."

    inner: Domain = dcls.field(default=AnyDomain(), kw_only=True)

    @property
    def __parents__(self) -> list[Domain]:
        return [SpatialDomain(SpatialDomain(self.inner))]

    def check_type(self, obj, /) -> None:
        super().check_type(obj)
        SpatialDomain(SpatialDomain(self.inner)).check_type(obj)

    def check_data(self, obj, /) -> None:
        super().check_data(obj)
        SpatialDomain(SpatialDomain(self.inner)).check_data(obj)


class SizedGridDomain(GridDomain):
    "A grid of this height and width."

    height: int
    width: int

    @property
    def __parents__(self) -> list[Domain]:
        return [GridDomain(inner=self.inner)]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if tuple(obj.shape[1:3]) != (self.height, self.width):
            raise InvalidDomain(
                f"size {tuple(obj.shape[1:3])}, expected {(self.height, self.width)}"
            )


class ImageDomain(GridDomain):
    "A grid of pixels, `[B, H, W, C]`, float pixels in [0, 1]."

    @property
    def __parents__(self) -> list[Domain]:
        return [GridDomain(inner=self.inner)]

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)
        if obj.is_floating_point() and not ((obj >= 0) & (obj <= 1)).all():
            raise InvalidDomain("float pixels must be in [0, 1]")


class SizedImageDomain(ImageDomain, SizedGridDomain):
    "An image of this height and width."

    @property
    def __parents__(self) -> list[Domain]:
        return [
            ImageDomain(inner=self.inner),
            SizedGridDomain(self.height, self.width, inner=self.inner),
        ]


# Videos ====


class VideoDomain(Domain):
    "Temporal[Image], `[B, T, H, W, C]`."

    inner: Domain = dcls.field(default=ImageDomain(), kw_only=True)

    @property
    def __parents__(self) -> list[Domain]:
        return [TemporalDomain(self.inner)]

    def check_type(self, obj, /) -> None:
        super().check_type(obj)
        TemporalDomain(self.inner).check_type(obj)

    def check_data(self, obj, /) -> None:
        super().check_data(obj)
        TemporalDomain(self.inner).check_data(obj)


class SizedVideoDomain(VideoDomain):
    "A video of this many frames."

    frames: int

    @property
    def __parents__(self) -> list[Domain]:
        return [VideoDomain(inner=self.inner)]

    def check_type(self, obj, /) -> None:
        super().check_type(obj)
        length = len(obj) if isinstance(obj, list | tuple) else obj.shape[1]
        if length != self.frames:
            raise InvalidDomain(f"{length} frames, expected {self.frames}")


# Utilities ====


def _items(obj) -> list:
    "The positions of a list, folded into the batch."

    if isinstance(obj, list | tuple):
        return list(obj)

    if len(getattr(obj, "shape", ())) < 2:
        raise InvalidDomain(f"a list is [B, L, ...], got {type(obj).__name__}")

    return [obj.flatten(0, 1)]
