# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn
from torchrl import data as tspec
from torchrl import modules as rlmods
from torchrl.data import tensor_specs as tspec

from aioway.spaces import Space, TSpecSpace

from ._utils import Activation, activation_class, activation_module
from .compound import BuilderNode, BuiltModule, CompoundBuilder
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = [
    "linear_regression",
    "ClfLogitHead",
    "MlpEmitter",
    "TorchRlMlpEmitter",
    "MlpCompoundEmitter",
]


@emitter_function
def linear_regression(observ: Space, action: Space) -> nn.Module:
    """
    `Linear` module from `ShapeSpace`s.
    """
    if not isinstance(observ, TSpecSpace):
        return NotImplemented
    if not isinstance(action, TSpecSpace):
        return NotImplemented

    if not isinstance(observ.spec, tspec.Unbounded):
        return NotImplemented

    if not isinstance(action.spec, tspec.Unbounded):
        return NotImplemented

    # The simple case where the `ndim` are all 1.
    if observ.ndim == action.ndim == 1:
        return nn.Linear(in_features=observ.shape[-1], out_features=action.shape[-1])

    module = nn.Sequential()

    # Flatten if it's not already.
    if observ.ndim != 1:
        module.append(nn.Flatten())

    module.append(
        nn.Linear(in_features=observ.shape.numel(), out_features=action.shape.numel())
    )

    # Map back.
    if action.ndim != 1:
        return module.append(nn.Unflatten(-1, action.shape))

    return module


@emitter_dcls
class ClfLogitHead(Emitter):
    "Emits a classification head."

    emitter: Emitter
    "Emits a regression model."

    def __call__(self, observ: Space, action: Space) -> nn.Module:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not isinstance(observ.spec, tspec.Unbounded):
            return NotImplemented

        if not isinstance(action.spec, tspec.BoundedDiscrete) or action.ndim != 0:
            return NotImplemented

        action_count = int(action.spec.high - action.spec.low + 1)

        module = nn.Sequential(
            self.emitter(observ, observ),
            nn.Flatten(),
            nn.Linear(observ.shape.numel(), action_count),
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

    activation: Activation = "relu"
    """
    The activation to use.
    """


@emitter_dcls
class TorchRlMlpEmitter(_MlpEmitter):
    "Emits a `torchrl.modules.MLP`."

    def __call__(self, observ: Space, action: Space) -> nn.Module:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        return rlmods.MLP(
            in_features=observ.shape[-1],
            out_features=action.shape[-1],
            num_cells=self.hidden_sizes,
            activation_class=activation_class(self.activation),
        )


@emitter_dcls
class MlpEmitter(_MlpEmitter):
    """
    Emits a `nn.Sequential` module.
    """

    def __call__(self, observ: Space, action: Space) -> nn.Sequential:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not isinstance(observ.spec, tspec.Unbounded):
            return NotImplemented

        if not isinstance(action.spec, tspec.Unbounded):
            return NotImplemented

        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]

        modules: list[nn.Module] = []
        activ = activation_module(self.activation)

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(in_features=in_feats, out_features=out_feats))

            if activ is not NotImplemented:
                modules.append(activ)

        # Drop the last one.
        if activ is not NotImplemented:
            modules.pop()

        return nn.Sequential(*modules)


@emitter_dcls
class MlpCompoundEmitter(_MlpEmitter):
    """
    Emits a simple MLP with hidden sizes and activation.
    """

    def __call__(self, observ: Space, action: Space) -> BuiltModule:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not isinstance(observ.spec, tspec.Unbounded):
            return NotImplemented

        if not isinstance(action.spec, tspec.Unbounded):
            return NotImplemented

        sizes = [observ.shape[-1], *self.hidden_sizes, action.shape[-1]]

        builder = CompoundBuilder()

        x: BuilderNode = builder.input("input")
        activ = activation_module(self.activation)

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            x = builder.thunk(
                nn.Linear(in_features=in_feats, out_features=out_feats),
                x,
            )

            if activ is not NotImplemented:
                x = builder.thunk(activ, x)

        return builder.output(x)
