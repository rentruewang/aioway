# Copyright (c) AIoWay Authors - All Rights Reserved

"The dataset related `Hop`s."

import copy
import typing

import torch

from aioway._streams import TensorStream

from .hop import Hop, hop_dcls

__all__ = ["TensorStreamHop"]


@hop_dcls
class TensorStreamHop(Hop):
    stream: TensorStream

    def forward(self) -> torch.Tensor:
        return next(self.stream)

    def _rebuild(self) -> typing.Self:
        return copy.copy(self)

    @property
    @typing.override
    def requires_grad(self) -> bool:
        return False
