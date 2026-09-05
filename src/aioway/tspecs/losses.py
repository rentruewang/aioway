# Copyright (c) AIoWay Authors - All Rights Reserved

"The `LossTSpec` interface."

import dataclasses as dcls
import typing

import torch
from torchrl.data import tensor_specs as tspecs

from .tspecs import TSpec

__all__ = ["LossTSpec"]


@typing.final
@dcls.dataclass(frozen=True)
class LossTSpec(TSpec):
    """
    The `TSpec` that will be marked as losses.
    """

    tspec: TSpec = dcls.field(default_factory=lambda: tspecs.Unbounded(torch.Size(())))

    def __post_init__(self) -> None:
        if self.tspec.shape:
            raise ValueError(f"{self.shape=} should be ().")

    def is_in(self, item):
        return self.tspec.is_in(item)

    @property
    def shape(self) -> torch.Size:
        return torch.Size(())

    def rand(self, batch: torch.Size, /):
        return self.tspec.rand(batch)
