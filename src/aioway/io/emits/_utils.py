# Copyright (c) AIoWay Authors - All Rights Reserved

import enum

import torch
from torch import nn

from aioway.io.tspecs import TSpecLike, as_tspec, sample_from_tspec, set_batch_size
from aioway.ir import NnProc, track_module_thunks
from aioway.t import TList, clone_in_fake_mode, fake_mode, route_aten_mode

__all__ = ["module_tspec_to_proc", "chain_sequential_proc", "Activation"]


def chain_sequential_proc(prev: NnProc, after: NnProc) -> NnProc:
    if not isinstance(outs := prev.outputs, TList) or len(outs) != 1:
        raise RuntimeError

    if len(after.inputs) != 1 or len(after.inputs.tensors) != 1:
        raise RuntimeError

    [prev_output_tensor] = outs
    [after_input_tensor] = after.inputs.tensors

    def swap_input(tensor: torch.Tensor) -> torch.Tensor:
        return prev_output_tensor if tensor is after_input_tensor else tensor

    new_list = [t.tree_map_only(torch.Tensor, swap_input) for t in after.instrs]
    return NnProc(
        instrs=[*prev.instrs, *new_list], inputs=prev.inputs, outputs=after.outputs
    )


def module_tspec_to_proc(module: nn.Module, *tspecs: TSpecLike) -> NnProc:
    """
    This is a convenient method to convert `(module, *tpsecs)` to an `NnProc`.

    This makes migration from `emit -> nn.Module` to `emit -> NnProc` easier.
    """

    with (
        fake_mode(),
        track_module_thunks() as thunks,
        route_aten_mode.activate(),
        clone_in_fake_mode(),
    ):
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
