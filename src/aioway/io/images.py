# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import os
import pathlib
import typing
from collections import abc as cabc

import torch
from torch import ops
from torchvision import io as vio

from aioway.fn import Preview, enabled_fake_mode, torch_real_mode

__all__ = ["read_image_from_path", "read_image_normalized"]


def read_image_from_path(fname: os.PathLike[str]):
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


def read_image_normalized(fname: os.PathLike[str]):
    return read_image_from_path(fname).float() / 255


@dcls.dataclass(frozen=True)
class ReadFile(Preview):
    IR = ops.image.read_file.default

    fname: str
    """
    The file name argument.
    """

    @typing.override
    def ok(self) -> bool:
        return True

    @typing.override
    def do(self) -> torch.Tensor:
        if not enabled_fake_mode():
            raise RuntimeError("Only works in fake mode!")

        with torch_real_mode():
            return self.IR(self.fname)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        return
        yield

    @typing.override
    def cost(self) -> int:
        return self.do().numel()


@dcls.dataclass(frozen=True)
class DecodeImage(Preview):
    IR = ops.image.decode_image.default

    input: torch.Tensor
    mode: vio.ImageReadMode
    apply_exif_orientation: bool

    @typing.override
    def ok(self) -> bool:
        return True

    @typing.override
    def do(self) -> torch.Tensor:
        if not enabled_fake_mode():
            raise RuntimeError("Only runs in fake mode!")

        with torch_real_mode():
            return self.IR(self.input, self.mode, self.apply_exif_orientation)

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield self.input

    @typing.override
    def cost(self) -> int:
        return self.do().numel()
