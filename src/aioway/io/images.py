# Copyright (c) AIoWay Authors - All Rights Reserved

import os
import pathlib

from torchvision import io as vio

from aioway.fn import torch_enable_fake_mode_func

__all__ = ["read_image_from_path", "read_image_normalized"]


@torch_enable_fake_mode_func(False)
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
