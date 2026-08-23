# Copyright (c) AIoWay Authors - All Rights Reserved

"Discover datasets."

import pathlib

from .dsets import Dset

__all__ = ["route_dset"]


def route_dset(name: str) -> Dset:
    # For MNIST we already have an adaptor.
    # Get rid of this at some point.
    if name.lower() == "mnist":
        from .mnist import MnistDataset

        return MnistDataset()

    # Not yet handled.
    if is_image_folder(name):
        raise NotImplementedError

    raise ValueError(f"Can't handle the input {name}. Give up.")


def is_image_folder(path: str | pathlib.Path) -> bool:
    path = pathlib.Path(path)

    if not path.exists():
        return False
    if not path.is_dir():
        return False
    if not path.is_dir():
        return False
    return True
