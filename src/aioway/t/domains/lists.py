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

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if not isinstance(obj, torch.Tensor):
            raise InvalidDomain(f"expected a tensor, got {type(obj).__name__}")


# Lists ====


class ListDomain(TensorDomain):
    "`inner` values in order, `[B, L, ...]`."

    inner: Domain = AnyDomain()

    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.ndim < 2:
            raise InvalidDomain(f"a list is [B, L, ...], got {obj.ndim} dims")
        self.inner.check_type(obj.flatten(0, 1))

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)
        self.inner.check_data(obj.flatten(0, 1))


class TemporalDomain(ListDomain):
    "A list in time, `[B, T, ...]`."

    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


class SpatialDomain(ListDomain):
    "A list in space, `[B, S, ...]`."

    @property
    def __extends__(self) -> list[Domain]:
        return [ListDomain(self.inner)]


# Grids ====


class GridDomain(TensorDomain):
    "Spatial[Spatial[inner]], `[B, H, W, ...]`."

    inner: Domain = dcls.field(default=AnyDomain(), kw_only=True)

    @property
    def __extends__(self) -> list[Domain]:
        return [SpatialDomain(SpatialDomain(self.inner))]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.ndim < 3:
            raise InvalidDomain(f"a grid is [B, H, W, ...], got {obj.ndim} dims")
        self.inner.check_type(obj.flatten(0, 2))

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)
        self.inner.check_data(obj.flatten(0, 2))


class SizedGridDomain(GridDomain):
    "A grid of this height and width."

    height: int
    width: int

    @property
    def __extends__(self) -> list[Domain]:
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
    def __extends__(self) -> list[Domain]:
        return [GridDomain(inner=self.inner)]

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)
        if obj.is_floating_point() and not ((obj >= 0) & (obj <= 1)).all():
            raise InvalidDomain("float pixels must be in [0, 1]")


class SizedImageDomain(ImageDomain, SizedGridDomain):
    "An image of this height and width."

    @property
    def __extends__(self) -> list[Domain]:
        return [
            ImageDomain(inner=self.inner),
            SizedGridDomain(self.height, self.width, inner=self.inner),
        ]


# Videos ====


class VideoDomain(TensorDomain):
    "Temporal[Image], `[B, T, H, W, C]`."

    inner: Domain = dcls.field(default=ImageDomain(), kw_only=True)

    @property
    def __extends__(self) -> list[Domain]:
        return [TemporalDomain(self.inner)]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.ndim < 2:
            raise InvalidDomain(f"a video is [B, T, ...], got {obj.ndim} dims")
        self.inner.check_type(obj.flatten(0, 1))

    def check_data(self, obj: torch.Tensor, /) -> None:
        super().check_data(obj)
        self.inner.check_data(obj.flatten(0, 1))


class SizedVideoDomain(VideoDomain):
    "A video of this many frames."

    frames: int

    @property
    def __extends__(self) -> list[Domain]:
        return [VideoDomain(inner=self.inner)]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.shape[1] != self.frames:
            raise InvalidDomain(f"{obj.shape[1]} frames, expected {self.frames}")
