# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Hop`s backed by torch tensors."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from .hop import Hop, hop_dcls

__all__ = ["TensorHop", "TensorListHop", "TensorDictHop"]


@hop_dcls
class TensorHop(Hop[torch.Tensor]):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tensor: torch.Tensor

    @typing.override
    def do(self) -> torch.Tensor:
        return self.tensor

    @typing.override
    def deps(self) -> cabc.Iterator[Hop]:
        return
        yield


@hop_dcls
class TensorListHop(Hop[list[torch.Tensor]]):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tensors: list[torch.Tensor]
    "The list of tensor that backs the `Hop`."

    @typing.override
    def do(self) -> list[torch.Tensor]:
        return self.tensors

    @typing.override
    def deps(self) -> cabc.Iterator[Hop]:
        return
        yield


@hop_dcls
class TensorDictHop(Hop[td.TensorDict]):
    """
    The `Hop` backed by a `torch.Tensor`.
    """

    tdict: td.TensorDict
    """
    The `td.TensorDict` that backs the `Hop`.
    """

    @typing.override
    def do(self) -> td.TensorDict:
        return self.tdict

    @typing.override
    def deps(self) -> cabc.Iterator[Hop]:
        return
        yield
