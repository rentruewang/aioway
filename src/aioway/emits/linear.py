# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn
from torchrl import data as rldata
from torchrl import modules as rlmods

from ._utils import Activation, activation_class, activation_module
from .compound import BuilderNode, BuiltModule, CompoundBuilder
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = ["linear_shape", "MlpEmitter", "TorchRlMlpEmitter", "MlpCompoundEmitter"]


@emitter_function
def linear_shape(observ: rldata.TensorSpec, action: rldata.TensorSpec) -> nn.Module:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observ, rldata.Unbounded):
        return NotImplemented

    if not isinstance(action, rldata.Unbounded):
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
        return nn.Unflatten(-1, action.shape)

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

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec
    ) -> nn.Module:

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

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec
    ) -> nn.Sequential:
        if not isinstance(observ, rldata.Unbounded):
            return NotImplemented

        if not isinstance(action, rldata.Unbounded):
            return NotImplemented

        sizes = [observ[-1], *self.hidden_sizes, action[-1]]

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

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec
    ) -> BuiltModule:
        if not isinstance(observ, rldata.Unbounded):
            return NotImplemented

        if not isinstance(action, rldata.Unbounded):
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
