# Copyright (c) AIoWay Authors - All Rights Reserved

"Normalization layers."

import typing

from torch import nn

from aioway.ops import OpFunc, SameShapeOp

from ._utils import observ_op
from .domains import FloatListDomain
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = ["NormEmitter", "NormType", "layer_norm_emitter"]

type NormType = typing.Literal["instance", "batch"]


@emitter_dcls
class NormEmitter(Emitter[FloatListDomain, FloatListDomain]):
    norm_type: NormType
    "The type of normalization layer."

    def emit(self, observ: FloatListDomain, action: FloatListDomain, /) -> OpFunc:
        if observ != action:
            return NotImplemented

        shape = observ.shape.torch()
        module = self._get_norm(len(shape) - 1, shape[0])
        return OpFunc(SameShapeOp(module, observ_op()))

    def _get_norm(self, non_channel_ndim: int, num_features: int) -> nn.Module:
        klass = getattr(nn, f"{self.norm_type.capitalize()}Norm{non_channel_ndim}d")
        return klass(num_features=num_features)


@emitter_function
def layer_norm_emitter(observ: FloatListDomain, action: FloatListDomain) -> OpFunc:
    if observ != action:
        return NotImplemented

    return OpFunc(SameShapeOp(nn.LayerNorm(observ.shape.torch()), observ_op()))
