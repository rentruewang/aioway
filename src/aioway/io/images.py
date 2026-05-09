# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch
from PIL import Image as image
from torchvision import io as vio
from torchvision.transforms import v2 as tt

from aioway.fn import enabled_fake_mode, torch_enable_fake_mode_func
from aioway.schemas import attr

__all__ = [
    "ImageLoader",
    "ComposedImageLoader",
    "PillowImageLoader",
    "FakePillowImageLoader",
    "TvioImageLoader",
]


@dcls.dataclass
class ImageLoader(abc.ABC):
    "The image loader API. Converts from a file name `fname` to a tensor."

    @abc.abstractmethod
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


@dcls.dataclass
class ComposedImageLoader(ImageLoader):
    """
    First load the image, then use `tt.Transform` to load it.

    It seems that `tt.Transform` uses mostly normal `torch.aten` operations,
    so we should have a good chance of making it work with fake mode.
    """

    loader: ImageLoader
    "The image loader."

    transform: tt.Transform = tt.Lambda(lambda t: t)
    """
    This is a series of `torchvision` `Transform`.
    Default to null op (a lambda because compose cannot be empty).
    """

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        tensor = self.loader(fname)
        return self.transform(tensor)


@dcls.dataclass
class PillowImageLoader(ImageLoader):
    """
    This is the pillow image loader + torchvision compose.
    """

    transform: tt.Transform = tt.PILToTensor()
    "The transform to use to convert the pillow image to a tensor."

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        img = image.open(fname)
        return self.transform(img)


@dcls.dataclass
class FakePillowImageLoader(ImageLoader):
    def __post_init__(self):
        if not enabled_fake_mode():
            raise RuntimeError(f"{type(self)} only works in fake mode!")

    @typing.override
    @torch_enable_fake_mode_func(True)
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        img = image.open(fname)

        return attr(
            {
                "shape": [len(img.mode), img.width, img.height],
                "device": "cpu",
                "dtype": "float32",
            }
        ).to_tensor()


@dcls.dataclass
class TvioImageLoader:
    """
    The loader backed by `torchvision.io`. It should be faster than the PIL route.

    But, it doesn't work in fake mode, so calling it always loads the tensor into memory.
    """

    norm: bool = True
    "If `True`, normalize the `uint8` tensor to 0-1 `float` tensor."

    def __call__(self, fname: str | pathlib.Path):
        reader = self._read_normalized if self.norm else self._read_image
        return reader(fname)

    @torch_enable_fake_mode_func(False)
    def _read_image(self, fname: str | pathlib.Path):
        fname_path = pathlib.Path(fname)
        fname_str = str(fname_path)

        match fname_path.suffix:
            # According to torch's documentation,
            # weirdly the `decode_image` function only decode these files.
            case ".jpg" | ".png" | ".gif" | ".webp":
                return vio.decode_image(fname_str)

            # This requires installing extra dependencies, do this later.
            # https://docs.pytorch.org/vision/main/generated/torchvision.io.decode_image.html
            case ".avif" | ".heic":
                raise ValueError(
                    "AVIF or HEIC requires extra dependencies, not supported yet!"
                )

            case _:
                raise ValueError(f"Does not support {fname_path.suffix} files.")

    def _read_normalized(self, fname: str | pathlib.Path):
        return self._read_image(fname).float() / 255
