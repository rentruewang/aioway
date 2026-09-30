# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import tensordict as td
import torch

__all__ = ["TensorId", "TdictId"]


class TensorId(int):
    """
    The id for `torch.Tensor`. Added for type safety.
    """

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, /) -> typing.Self:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(type(tensor))

        return cls(id(tensor))


class TdictId(int):
    """
    The id for `TensorDict`'s tensor collection. Added for type safety.
    """

    @classmethod
    def from_tdict(cls, t: td.TensorDictBase, /) -> typing.Self:
        if not td.is_tensor_collection(t):
            raise TypeError(type(t))

        return cls(id(t))
