# Copyright (c) AIoWay Authors - All Rights Reserved

"Normalization layers."

import typing

from torch import nn
from torchrl import data as rldata

from .emitters import Emitter, emitter_dcls

__all__ = ["NormEmitter", "NormType"]

type NormType = typing.Literal["instance", "batch"]


@emitter_dcls
class NormEmitter(Emitter):
    norm_type: NormType
    "The type of normalization layer."

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec
    ) -> nn.Module:
        if not isinstance(observ, rldata.Unbounded):
            return NotImplemented

        if not isinstance(action, rldata.Unbounded):
            return NotImplemented

        if observ != action:
            return NotImplemented

        return self._get_norm(observ.ndim - 1, observ.shape[0])

    def _get_norm(self, non_channel_ndim: int, num_features: int) -> nn.Module:
        klass = getattr(nn, f"{self.norm_type.capitalize()}Norm{non_channel_ndim}d")
        return klass(num_features=num_features)
