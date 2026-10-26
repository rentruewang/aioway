# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.t import route_aten_thunk
import enum

from torch import nn

from aioway.cc.tspecs import TSpecLike, as_tspec, sample_from_tspec, set_batch_size
from aioway.ir import NnProc, track_module_thunks
from aioway.t import fake_mode

__all__ = ["module_tspec_to_proc", "Activation"]


def module_tspec_to_proc(module: nn.Module, *tspecs: TSpecLike) -> NnProc:
    """
    This is a convenient method to convert `(module, *tpsecs)` to an `NnProc`.

    This makes migration from `emit -> nn.Module` to `emit -> NnProc` easier.
    """

    with track_module_thunks() as thunks, fake_mode(), route_aten_thunk.activate():
        with set_batch_size(1):
            inputs = tuple(sample_from_tspec(as_tspec(t)) for t in tspecs)

        _ = module(*inputs)

    return thunks.procedure


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
