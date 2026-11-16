# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.io.tspecs import TSpec
from aioway.ops import LayerOp, OpFunc, SameShapeOp, ViewOp
from torch import nn
from torchrl import modules as rlmods
from torchrl.data import tensor_specs as tspecs

from ._utils import Activation, linear_tree, observ_op
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = [
    "linear_regression",
    "ClfLogitHead",
    "MlpEmitter",
    "TorchRlMlpEmitter",
]


@emitter_function
def linear_regression(observ: TSpec, action: TSpec) -> OpFunc:
    """
    `Linear` from `Unbounded` to `Unbounded`.
    """

    if not isinstance(observ, tspecs.Unbounded):
        return NotImplemented

    if not isinstance(action, tspecs.Unbounded):
        return NotImplemented

    return OpFunc(linear_tree(observ_op(), observ.shape, action.shape))


@emitter_dcls
class ClfLogitHead(Emitter):
    "Emits a classification head."

    emitter: Emitter
    "Emits a regression model."

    def __call__(self, observ: TSpec, action: TSpec) -> OpFunc:

        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.BoundedDiscrete):
            return NotImplemented

        first = self.emitter(observ, observ)

        if first is NotImplemented:
            return NotImplemented

        action_count = int(action.high - action.low + 1)

        flat = ViewOp(nn.Flatten(), first.output)
        return OpFunc(LayerOp(nn.Linear(observ.shape.numel(), action_count), flat))


@emitter_dcls
class _MlpEmitter(Emitter):
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


@emitter_dcls
class TorchRlMlpEmitter(_MlpEmitter):
    "Emits a `torchrl.modules.MLP`."

    def __call__(self, observ: TSpec, action: TSpec) -> OpFunc:
        module = rlmods.MLP(
            in_features=observ.shape[-1],
            out_features=action.shape[-1],
            num_cells=self.hidden_sizes,
            activation_class=Activation(self.activation).nn_type,
        )
        return OpFunc(LayerOp(module, observ_op()))


@emitter_dcls
class MlpEmitter(_MlpEmitter):
    """
    Emits `Linear`s with activations in between.
    """

    def __call__(self, observ: TSpec, action: TSpec) -> OpFunc:
        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.Unbounded):
            return NotImplemented

        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]
        activation = Activation(self.activation).nn_type

        op = observ_op()

        for i, (in_feats, out_feats) in enumerate(zip(sizes[:-1], sizes[1:])):
            # Activations only in between `Linear`s.
            if i:
                op = SameShapeOp(activation(), op)

            op = LayerOp(nn.Linear(in_features=in_feats, out_features=out_feats), op)

        return OpFunc(op)
