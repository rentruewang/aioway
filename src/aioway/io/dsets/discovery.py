# Copyright (c) AIoWay Authors - All Rights Reserved

"Discover datasets."
import pathlib

import datasets

from .dsets import Dset

__all__ = ["route_dset"]

dset = datasets.load_dataset("hi")
train = dset["train"]

train.
