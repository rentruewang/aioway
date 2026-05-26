# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Hop`s backed by torch tensors."

import typing

import tensordict as td
import torch

from aioway._fn import thunk_dcls

from .hop import HopFwd, HopInit, hop_init_dcls

__all__ = ["TensorHopInit", "TensorListHopInit", "TensorDictHopInit"]


@thunk_dcls
class TensorHopFwd[T](HopFwd):
    """
    The `HopFwd` subclass for `torch.Tensor`s.
    """

    data: T

    @typing.override
    def fwd(self) -> T:
        return self.data


@hop_init_dcls
class TensorHopInit(HopInit):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tensor: torch.Tensor

    @typing.override
    def init(self):
        return TensorHopFwd(self.tensor)


@hop_init_dcls
class TensorListHopInit(HopInit):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tensors: list[torch.Tensor]
    "The list of tensor that backs the `Hop`."

    @typing.override
    def init(self):
        return TensorHopFwd(self.tensors)


@hop_init_dcls
class TensorDictHopInit(HopInit):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tdict: td.TensorDict
    """
    The `td.TensorDict` that backs the `Hop`.
    """

    @typing.override
    def init(self):
        return TensorHopFwd[td.TensorDict](self.tdict)
