# Copyright (c) AIoWay Authors - All Rights Reserved

from torchrl.data import tensor_specs as tspecs

from aioway.instrs import (
    Activation,
    Flatten,
    Instr,
    Linear,
    RlMlp,
    Sequential,
    Unflatten,
)
from aioway.tspecs import TSpec

from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = [
    "linear_regression",
    "ClfLogitHead",
    "MlpEmitter",
    "TorchRlMlpEmitter",
]


@emitter_function
def linear_regression(observ: TSpec, action: TSpec) -> Instr:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observ, tspecs.Unbounded):
        return NotImplemented

    if not isinstance(action, tspecs.Unbounded):
        return NotImplemented

    # The simple case where the `ndim` are all 1.
    if observ.ndim == action.ndim == 1:
        return Linear(in_features=observ.shape[-1], out_features=action.shape[-1])

    module_list: list[Instr] = []

    # Flatten if it's not already.
    if observ.ndim != 1:
        module_list.append(Flatten())

    module_list.append(
        Linear(in_features=observ.shape.numel(), out_features=action.shape.numel())
    )

    # Map back.
    if action.ndim != 1:
        module_list.append(Unflatten(-1, action.shape))

    return Sequential(*module_list)


@emitter_dcls
class ClfLogitHead(Emitter):
    "Emits a classification head."

    emitter: Emitter
    "Emits a regression model."

    def __call__(self, observ: TSpec, action: TSpec) -> Instr:

        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.BoundedDiscrete) or action.ndim != 0:
            return NotImplemented

        action_count = int(action.high - action.low + 1)

        module = Sequential(
            self.emitter(observ, observ),
            Flatten(),
            Linear(observ.shape.numel(), action_count),
        )

        return module


@emitter_dcls
class _MlpEmitter(Emitter):
    """
    Emits a `nn.Sequential` module.
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

    def __call__(self, observ: TSpec, action: TSpec) -> RlMlp:
        return RlMlp(
            in_features=observ.shape[-1],
            out_features=action.shape[-1],
            num_cells=self.hidden_sizes,
            activation_class=Activation(self.activation),
        )


@emitter_dcls
class MlpEmitter(_MlpEmitter):
    """
    Emits a `nn.Sequential` module.
    """

    def __call__(self, observ: TSpec, action: TSpec) -> Sequential:
        if not isinstance(observ, tspecs.Unbounded):
            return NotImplemented

        if not isinstance(action, tspecs.Unbounded):
            return NotImplemented

        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]

        modules: list[Instr] = []
        activ = Activation(self.activation).instr_cls()

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            modules.append(Linear(in_features=in_feats, out_features=out_feats))

            if activ is not NotImplemented:
                modules.append(activ)

        # Drop the last one.
        if activ is not NotImplemented:
            modules.pop()

        return Sequential(*modules)
