# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of tensors: grids, images, videos."

import torch
from torchvision import tv_tensors as tvt

from .domains import Domain, InvalidDomainValue

__all__ = [
    "TensorDomain",
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
            raise InvalidDomainValue(f"expected a tensor, got {type(obj).__name__}")


class GridDomain(TensorDomain):
    "A lattice, torchvision's layout: [..., C, H, W]."

    @property
    def __parents__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.ndim < 3:
            raise InvalidDomainValue(f"a grid is [..., C, H, W], got {obj.ndim} dims")


class SizedGridDomain(GridDomain):
    "A grid of this height and width."

    height: int
    """
    The number of rows, `H` in `[..., C, H, W]`.
    """

    width: int
    """
    The number of columns, `W` in `[..., C, H, W]`.
    """

    @property
    def __parents__(self) -> list[Domain]:
        return [GridDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        hw = tuple(obj.shape[-2:])
        if hw != (self.height, self.width):
            raise InvalidDomainValue(f"size {hw}, expected {(self.height, self.width)}")


class ImageDomain(GridDomain):
    "A grid that is a torchvision `Image`, float pixels in [0, 1]. A fake cannot tell it from a grid."

    @property
    def __parents__(self) -> list[Domain]:
        return [GridDomain()]

    def check_data(self, obj: torch.Tensor) -> None:
        super().check_data(obj)
        if not isinstance(obj, tvt.Image):
            raise InvalidDomainValue("expected a torchvision Image")
        if not _in_unit(obj):
            raise InvalidDomainValue("float pixels must be in [0, 1]")


class SizedImageDomain(ImageDomain, SizedGridDomain):
    "An image of this height and width."

    @property
    def __parents__(self) -> list[Domain]:
        return [ImageDomain(), SizedGridDomain(self.height, self.width)]


class VideoDomain(TensorDomain):
    "Seq[Image]: a torchvision `Video`, [..., T, C, H, W], each frame a grid."

    @property
    def __parents__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.ndim < 4:
            raise InvalidDomainValue(
                f"a video is [..., T, C, H, W], got {obj.ndim} dims"
            )
        GridDomain().check_type(obj.select(-4, 0))

    def check_data(self, obj: torch.Tensor) -> None:
        super().check_data(obj)

        if not isinstance(obj, tvt.Video):
            raise InvalidDomainValue("expected a torchvision Video")
        if not _in_unit(obj):
            raise InvalidDomainValue("float pixels must be in [0, 1]")


class SizedVideoDomain(VideoDomain):
    "A video of this many frames."

    frames: int
    """
    The number of frames, `T` in `[..., T, C, H, W]`.
    """

    @property
    def __parents__(self) -> list[Domain]:
        return [VideoDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)

        if obj.shape[-4] != self.frames:
            raise InvalidDomainValue(f"{obj.shape[-4]} frames, expected {self.frames}")


# Utilities ====


def _in_unit(x: torch.Tensor) -> bool:
    "A float image holds values in [0, 1]."

    return not x.is_floating_point() or bool(((x >= 0) & (x <= 1)).all())
