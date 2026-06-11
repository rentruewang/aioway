# Copyright (c) AIoWay Authors - All Rights Reserved

"The dataset related `Hop`s."

import typing
from collections import abc as cabc

import torch

from .hop import TensorHop, hop_dcls

__all__ = ["TensorListHop"]


@hop_dcls
class TensorListHop(TensorHop):
    "A `Hop` backed by a list of `torch.Tensor`."

    sequence: cabc.Sequence[torch.Tensor]
    "List of `torch.Tensor`s."

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    def iterate(self):
        for batch in self.sequence:
            yield batch
