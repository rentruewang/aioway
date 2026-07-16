# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._ufuncs import BuilderNode, CompoundBuilder, UFunc
from aioway.emits import Emitter, emitter_dcls
from aioway.spaces import AttrSpace, ShapeSpace, Space
from aioway.torch.nn import CELU, GELU, Linear, NnInit, ReLU, ReLU6, Sigmoid, Tanh

from .emitters import FuncEmitter

__all__ = ["linear_shape", "linear_from_attr", "MlpEmitter"]


@FuncEmitter
def linear_shape(observation_space: Space, action_space: Space) -> UFunc:
    """
    `Linear` module from `ShapeSpace`s.
    """

    if not isinstance(observation_space, ShapeSpace):
        return NotImplemented

    if not isinstance(action_space, ShapeSpace):
        return NotImplemented

    return Linear(
        in_features=observation_space[-1], out_features=action_space[-1]
    ).ufunc


@FuncEmitter
def linear_from_attr(observation_space: Space, action_space: Space) -> UFunc:
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

    return Linear(in_features=observ_shape[-1], out_features=act_shape[-1]).ufunc


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
                Linear(in_features=in_feats, out_features=out_feats).ufunc, x
            )

            if activ is not NotImplemented:
                x = builder.thunk(activ, x)

        return builder.output(x)

    @property
    def _nn_init(self) -> NnInit:
        match self.activation:
            case None:
                return NotImplemented
            case "relu":
                return ReLU()
            case "relu6":
                return ReLU6()
            case "celu":
                return CELU()
            case "gelu":
                return GELU()
            case "sigmoid":
                return Sigmoid()
            case "tanh":
                return Tanh()

    @property
    def _activ_ufunc(self) -> UFunc:
        nn_init = self._nn_init
        if nn_init is NotImplemented:
            return NotImplemented
        else:
            return nn_init.ufunc
