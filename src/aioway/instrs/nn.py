# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._utils import dcls_asdict, render_fcall
from aioway.dsets import InputTarget
from aioway.modes import fake_mode
from aioway.tspecs import TSpecInfer, TSpecLike

from .instrs import AiowayModule, Instr

__all__ = ["NnInstr", "instr_dcls", "NnLayer", "NnLoss", "UnboundedInfer"]


@typing.dataclass_transform()
@typing.no_type_check
def instr_dcls(cls):
    "Decorator of dataclass for `Instr`."
    return dcls.dataclass(repr=False)(cls)


class NnLayer(AiowayModule[torch.Tensor, torch.Tensor]):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def __repr__(self) -> str:
        return self.__repr

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layer(tensor)

    @functools.cached_property
    def __repr(self) -> str:
        return "nn_layer::" + repr(self.loss)


class NnLoss(AiowayModule[InputTarget, torch.Tensor]):
    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self.loss = loss

    def __repr__(self) -> str:
        return self.__repr

    def forward(self, input_target: InputTarget) -> torch.Tensor:
        return self.loss(input_target.input, input_target.target)

    @functools.cached_property
    def __repr(self) -> str:
        return "nn_loss::" + repr(self.loss)


@dcls.dataclass(frozen=True)
class UnboundedInfer(TSpecInfer):
    """
    Infer an unbounded output from an unbounded input.
    """

    instr: Instr

    def __call__(self, tspec: TSpecLike, /) -> tspecs.Unbounded:
        assert isinstance(tspec, tspecs.Unbounded)

        with fake_mode():
            module = self.instr.module()
            random = tspec.sample(torch.Size([1]))

        output = module(random)
        return tspecs.Unbounded(shape=output.shape)


@instr_dcls
class NnInstr(Instr, abc.ABC):
    """
    `NnInstr` records the signature of an `nn.Module` initialization, and creates it.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls_asdict(self))

    @typing.override
    def module(self) -> AiowayModule:
        return NnLayer(self.NN(**dcls_asdict(self)))

    @typing.override
    def children(self):
        return ()
