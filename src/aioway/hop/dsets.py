# Copyright (c) AIoWay Authors - All Rights Reserved

"The dataset related `Hop`s."

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

    @property
    @typing.override
    def requires_grad(self) -> bool:
        return False
