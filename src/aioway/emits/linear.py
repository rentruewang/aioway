# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway.spaces import AttrSpace, ShapeSpace, Space

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


type Activation = typing.Literal[
    None, "relu", "relu6", "celu", "gelu", "sigmoid", "tanh"
]


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

        module = nn.Sequential()
        activ = _activ_module(self.activation)

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            module.append(nn.Linear(in_features=in_feats, out_features=out_feats))

            if activ is not NotImplemented:
                module.append(activ)

        return module


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
        activ = _activ_module(self.activation)

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            x = builder.thunk(
                nn.Linear(in_features=in_feats, out_features=out_feats),
                x,
            )

            if activ is not NotImplemented:
                x = builder.thunk(activ, x)

        return builder.output(x)


def _activ_module(activation: Activation) -> nn.Module:
    match activation:
        case None:
            return NotImplemented
        case "relu":
            return nn.ReLU()
        case "relu6":
            return nn.ReLU6()
        case "celu":
            return nn.CELU()
        case "gelu":
            return nn.GELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
