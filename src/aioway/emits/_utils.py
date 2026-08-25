# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

__all__ = ["Activation", "activation_module", "activation_class"]

type Activation = typing.Literal[
    None, "relu", "relu6", "celu", "gelu", "sigmoid", "tanh"
]


def activation_class(activation: Activation) -> type[nn.Module]:
    match activation:
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


def activation_module(activation: Activation) -> nn.Module:
    klass = activation_class(activation)

    if klass is NotImplemented:
        return NotImplemented

    return klass()
