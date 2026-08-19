# Copyright (c) AIoWay Authors - All Rights Reserved

"Normalization layers."

import typing

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.spaces import Space, TSpecSpace

from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = ["NormEmitter", "NormType", "layer_norm_emitter"]

type NormType = typing.Literal["instance", "batch"]


@emitter_dcls
class NormEmitter(Emitter):
    norm_type: NormType
    "The type of normalization layer."

    def __call__(self, observ: Space, action: Space) -> nn.Module:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not observ.cast_spec(tspecs.Unbounded):
            return NotImplemented

        if not action.cast_spec(tspecs.Unbounded):
            return NotImplemented

        if observ != action:
            return NotImplemented

        return self._get_norm(observ.ndim - 1, observ.shape[0])

    def _get_norm(self, non_channel_ndim: int, num_features: int) -> nn.Module:
        klass = getattr(nn, f"{self.norm_type.capitalize()}Norm{non_channel_ndim}d")
        return klass(num_features=num_features)


@emitter_function
def layer_norm_emitter(observ, action):
    if not isinstance(observ, TSpecSpace):
        return NotImplemented
    if not isinstance(action, TSpecSpace):
        return NotImplemented

    if not observ.cast_spec(tspecs.Unbounded):
        return NotImplemented

    if not action.cast_spec(tspecs.Unbounded):
        return NotImplemented

    if observ != action:
        return NotImplemented

    return nn.LayerNorm(observ.shape)
