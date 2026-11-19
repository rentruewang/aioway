# Copyright (c) AIoWay Authors - All Rights Reserved

"Normalization layers."

import typing

import torch
from torch import nn
from aioway.ops import OpFunc, SameShapeOp
from aioway.t import Domain

from ._utils import observ_op
from .domains import FloatListDomain
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = ["NormEmitter", "NormType", "layer_norm_emitter"]

type NormType = typing.Literal["instance", "batch"]


@emitter_dcls
class NormEmitter(Emitter):
    norm_type: NormType
    "The type of normalization layer."

    def __call__(self, observ: Domain, action: Domain) -> OpFunc:

        if not isinstance(observ, FloatListDomain):
            return NotImplemented

        if not isinstance(action, FloatListDomain):
            return NotImplemented

        if observ != action:
            return NotImplemented

        shape = torch.Size(observ.shape)
        module = self._get_norm(len(shape) - 1, shape[0])
        return OpFunc(SameShapeOp(module, observ_op()))

    def _get_norm(self, non_channel_ndim: int, num_features: int) -> nn.Module:
        klass = getattr(nn, f"{self.norm_type.capitalize()}Norm{non_channel_ndim}d")
        return klass(num_features=num_features)


@emitter_function
def layer_norm_emitter(observ: Domain, action: Domain) -> OpFunc:

    if not isinstance(observ, FloatListDomain):
        return NotImplemented

    if not isinstance(action, FloatListDomain):
        return NotImplemented

    if observ != action:
        return NotImplemented

    return OpFunc(SameShapeOp(nn.LayerNorm(torch.Size(observ.shape)), observ_op()))
