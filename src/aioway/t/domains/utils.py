# Copyright (c) AIoWay Authors - All Rights Reserved

"Utilities to build tensor domains."

from .domains import AnyDomain, Domain
from .tensors import (
    ListDomain,
    SizedListDomain,
    SizedMixin,
    SizedSpatialDomain,
    SizedTemporalDomain,
    SpatialDomain,
    TemporalDomain,
    TensorItemDomain,
)

__all__ = [
    "sized_of",
    "grid_of",
    "image",
    "sized_image",
    "video",
    "sized_video",
]


def sized_of(*shape: int) -> SizedMixin:
    "Nested sized lists of `shape`, ending with an item."

    domain: SizedMixin = TensorItemDomain()

    for length in reversed(shape):
        domain = SizedListDomain(domain, length=length)

    return domain


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
