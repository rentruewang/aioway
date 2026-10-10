# Copyright (c) AIoWay Authors - All Rights Reserved

import enum

from torch import nn


class Activation(enum.StrEnum):
    """
    The activation enum.
    """

    RELU = "relu"
    RELU6 = "relu6"
    CELU = "celu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    @property
    def nn_type(self) -> type[nn.Module]:
        "Convert from the `Activation` to `nn.Module` type."

        match self:
            case self.RELU:
                return nn.ReLU
            case self.RELU6:
                return nn.ReLU6
            case self.CELU:
                return nn.CELU
            case self.GELU:
                return nn.GELU
            case self.SIGMOID:
                return nn.Sigmoid
            case self.TANH:
                return nn.Tanh

        raise NotImplementedError(self)
