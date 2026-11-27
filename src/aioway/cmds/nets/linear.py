# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway.ops import LayerOp, OpFunc, SameShapeOp, ViewOp

from ._utils import Activation, linear_tree, observ_op
from .domains import FloatListDomain, IntRangeItemDomain
from .emitters import Emitter, EmitterLike, emitter_function

__all__ = [
    "linear_regression",
    "ClfLogitHead",
    "MlpEmitter",
]


@emitter_function
def linear_regression(observ: FloatListDomain, action: FloatListDomain) -> OpFunc:
    """
    `Linear` from float to float.
    """

    in_shape, out_shape = observ.shape.torch(), action.shape.torch()
    return OpFunc(linear_tree(observ_op(), in_shape, out_shape))


class ClfLogitHead(Emitter[FloatListDomain, IntRangeItemDomain]):
    "Emits a classification head."

    emitter: EmitterLike
    "Emits a regression model."

    def emit(self, observ: FloatListDomain, action: IntRangeItemDomain, /) -> OpFunc:
        first = self.emitter(observ, observ)

        if first is NotImplemented:
            return NotImplemented

        action_count = int(action.high - action.low + 1)

        flat = ViewOp(nn.Flatten(), first.output)
        in_features = observ.shape.numel()
        return OpFunc(LayerOp(nn.Linear(in_features, action_count), flat))


class _MlpEmitter(Emitter[FloatListDomain, FloatListDomain]):
    """
    Emits an MLP.
    """

    hidden_sizes: list[int]
    """
    The hidden sizes of MLP.
    """

    activation: str = "relu"
    """
    The activation to use.
    """


class MlpEmitter(_MlpEmitter):
    """
    Emits `Linear`s with activations in between.
    """

    def emit(self, observ: FloatListDomain, action: FloatListDomain, /) -> OpFunc:
        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]
        activation = Activation(self.activation).nn_type

        op = observ_op()

        for i, (in_feats, out_feats) in enumerate(zip(sizes[:-1], sizes[1:])):
            # Activations only in between `Linear`s.
            if i:
                op = SameShapeOp(activation(), op)

            op = LayerOp(nn.Linear(in_features=in_feats, out_features=out_feats), op)

        return OpFunc(op)
