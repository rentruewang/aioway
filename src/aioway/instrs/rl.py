# Copyright (c) AIoWay Authors - All Rights Reserved

import enum
import typing
from collections import abc as cabc

from torchrl import modules as rlmods

from aioway.instrs import CELU, GELU, Instr, ReLU, ReLU6, Sigmoid, Tanh

from .nn import instr_dcls

__all__ = ["Activation", "RlMlp"]


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
    def instr_cls(self) -> type[Instr]:
        "Convert from the `Activation` to `Instr` type."

        match self:
            case self.RELU:
                return ReLU
            case self.RELU6:
                return ReLU6
            case self.CELU:
                return CELU
            case self.GELU:
                return GELU
            case self.SIGMOID:
                return Sigmoid
            case self.TANH:
                return Tanh

        raise NotImplementedError(self)


@typing.final
@instr_dcls
class RlMlp(Instr):
    in_features: int
    out_features: int
    num_cells: cabc.Sequence[int]
    activation_class: Activation

    @typing.override
    def module(self) -> rlmods.MLP:
        return rlmods.MLP(
            in_features=self.in_features,
            out_features=self.out_features,
            num_cells=self.num_cells,
            activation_class=self.activation_class.instr_cls().module(),
        )

    def children(self):
        return ()
