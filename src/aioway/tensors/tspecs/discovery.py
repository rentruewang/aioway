# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torch.utils import data as dutils

from .tspecs import TSpec

__all__ = ["tensor_tspec", "tensordict_tspec", "tensorclass_tspec", "dataloader_tspec"]


def tensor_tspec(tensor: torch.Tensor) -> TSpec:
    raise NotImplementedError


def tensordict_tspec(tensor: td.TensorDict) -> TSpec:
    raise NotImplementedError


def tensorclass_tspec(tensor: type) -> TSpec:
    if not isinstance(tensor, type):
        raise TypeError(f"Only accepts types. Got {tensor=}.")

    raise NotImplementedError


def dataloader_tspec(dset: dutils.DataLoader) -> TSpec:
    raise NotImplementedError
