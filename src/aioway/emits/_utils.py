# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

__all__ = ["Activation", "activation_module"]

type Activation = typing.Literal[
    None, "relu", "relu6", "celu", "gelu", "sigmoid", "tanh"
]


def activation_module(activation: Activation) -> nn.Module:
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
