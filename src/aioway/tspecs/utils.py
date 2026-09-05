# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import typing
from collections import abc as cabc

import torch
from torchrl.data import tensor_specs as tspecs

from .tspecs import TSpec

__all__ = ["sample_from_tspec", "set_batch_size", "ArgsTSpec"]

_batch_size: torch.Size | None = None
"The batch size to use for emitting."


@ctxl.contextmanager
def set_batch_size(*batch_size: int) -> cabc.Generator[None]:
    "Configure the batch size to use with `spec` and `emit`."

    global _batch_size
    _batch_size = torch.Size(batch_size)

    try:
        yield
    finally:
        _batch_size = None


def sample_from_tspec(spec: TSpec, /) -> typing.Any:
    "Sample from the `spec` with the batch size configured by `with_batch_size`."

    assert _batch_size
    return spec.sample(torch.Size(_batch_size))


class ArgsTSpec(tspecs.Composite):
    """
    The `Composite` subclass that represents an argument list.
    """
