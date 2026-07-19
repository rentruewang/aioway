# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._ufuncs import BuilderNode, CompoundBuilder, UFunc
from aioway.emits import Emitter, emitter_dcls
from aioway.spaces import AttrSpace, ShapeSpace, Space
from aioway.torch.nn import NnLayerUFunc, NnUFunc, nn_ufunc

from .emitters import FuncEmitter

__all__ = ["linear_shape", "linear_from_attr", "MlpEmitter"]


@FuncEmitter
def linear_shape(observation_space: Space, action_space: Space) -> NnUFunc:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observation_space, ShapeSpace):
        return NotImplemented

    if not isinstance(action_space, ShapeSpace):
        return NotImplemented

    return nn_ufunc(
        nn.Linear, in_features=observation_space[-1], out_features=action_space[-1]
    )


@FuncEmitter
def linear_from_attr(observation_space: Space, action_space: Space) -> NnUFunc:
    """
    `Linear` module from `AttrShape`s.
    """

    if not isinstance(observation_space, AttrSpace):
        return NotImplemented

    if not isinstance(action_space, AttrSpace):
        return NotImplemented

    if (observ_shape := observation_space.shape) is None:
        return NotImplemented
    if (act_shape := action_space.shape) is None:
        return NotImplemented

    return nn_ufunc(nn.Linear, in_features=observ_shape[-1], out_features=act_shape[-1])


type Activation = typing.Literal[
    None, "relu", "relu6", "celu", "gelu", "sigmoid", "tanh"
]


@emitter_dcls
class MlpEmitter(Emitter):
    """
    Emits a simple MLP with hidden sizes and activation.
    """

    hidden_sizes: list[int]
    """
    The hidden sizes of MLP.
    """

    activation: Activation = "relu"
    """
    The activation to use.
    """

    def __call__(self, observation_space: Space, action_space: Space) -> UFunc:
        if not isinstance(observation_space, ShapeSpace):
            return NotImplemented

        if not isinstance(action_space, ShapeSpace):
            return NotImplemented

        sizes = [observation_space[-1], *self.hidden_sizes, action_space[-1]]

        builder = CompoundBuilder()

        x: BuilderNode = builder.input("input")
        activ = self._activ_ufunc

        for in_feats, out_feats in zip(sizes[:-1], sizes[1:]):
            x = builder.thunk(
                nn_ufunc(nn.Linear, in_features=in_feats, out_features=out_feats),
                x,
            )

            if activ is not NotImplemented:
                x = builder.thunk(activ, x)

        return builder.output(x)

    @property
    def _nn_init(self) -> type[nn.Module]:
        match self.activation:
            case None:
                return NotImplemented
            case "relu":
                return nn.ReLU
            case "relu6":
                return nn.ReLU6
            case "celu":
                return nn.CELU
            case "gelu":
                return nn.GELU
            case "sigmoid":
                return nn.Sigmoid
            case "tanh":
                return nn.Tanh

    @property
    def _activ_ufunc(self) -> UFunc:
        if self._nn_init is NotImplemented:
            return NotImplemented

        return NnLayerUFunc(self._nn_init)
