# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from aioway.spaces import AttrSpace, ShapeSpace, Space

from ._utils import Activation, activation_module
from .compound import BuilderNode, BuiltModule, CompoundBuilder
from .emitters import Emitter, emitter_dcls, emitter_function

__all__ = ["linear_shape", "linear_from_attr", "MlpEmitter", "MlpCompoundEmitter"]


@emitter_function
def linear_shape(observ: Space, action: Space) -> nn.Linear:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observ, ShapeSpace):
        return NotImplemented

    if not isinstance(action, ShapeSpace):
        return NotImplemented

    return nn.Linear(in_features=observ[-1], out_features=action[-1])


@emitter_function
def linear_from_attr(observ: Space, action: Space) -> nn.Module:
    """
    `Linear` module from `AttrShape`s.
    """

    if not isinstance(observ, AttrSpace):
        return NotImplemented

    if not isinstance(action, AttrSpace):
        return NotImplemented

    if (observ_shape := observ.shape) is None:
        return NotImplemented
    if (act_shape := action.shape) is None:
        return NotImplemented

    return nn.Linear(in_features=observ_shape[-1], out_features=act_shape[-1])


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
class MlpEmitter(_MlpEmitter):
    """
    Emits a `nn.Sequential` module.
    """

    def __call__(self, observ: Space, action: Space) -> nn.Sequential:
        if not isinstance(observ, ShapeSpace):
            return NotImplemented

        if not isinstance(action, ShapeSpace):
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

    def __call__(self, observ: Space, action: Space) -> BuiltModule:
        if not isinstance(observ, ShapeSpace):
            return NotImplemented

        if not isinstance(action, ShapeSpace):
            return NotImplemented

        sizes = [observ[-1], *self.hidden_sizes, action[-1]]

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
