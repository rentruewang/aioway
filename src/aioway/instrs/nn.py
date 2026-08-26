# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._utils import dcls_asdict, render_fcall
from aioway.tspecs import TSpecInfer

from .instrs import Instr

__all__ = ["NnInstr", "nn_instr_dcls"]


@typing.dataclass_transform()
@typing.no_type_check
def nn_instr_dcls(cls):
    "Decorator of dataclass for `NnInstr`."
    return dcls.dataclass(repr=False)(cls)


@nn_instr_dcls
class NnInstr(Instr, abc.ABC):
    """
    `NnInstr` records the signature of an `nn.Module` initialization, and creates it.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls_asdict(self))

    @typing.override
    def __tspec_infer__(self) -> TSpecInfer:
        raise NotImplementedError

    @typing.override
    def module(self) -> nn.Module:
        return self.NN(**dcls_asdict(self))

    @typing.override
    def children(self):
        return ()

    @classmethod
    @typing.override
    def _lift(cls, module: nn.Module) -> typing.Self:
        raise NotImplementedError
