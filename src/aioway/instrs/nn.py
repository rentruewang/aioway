# Copyright (c) AIoWay Authors - All Rights Reserved

"The base class for `NnInfer`, bring `nn.Module` by `torch` into `Instr`."

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._utils import dcls_asdict, render_fcall

from .instrs import Instr

__all__ = ["NnInstr", "instr_dcls"]


@typing.dataclass_transform()
@typing.no_type_check
def instr_dcls(cls):
    "Decorator of dataclass for `Instr`."
    return dcls.dataclass(repr=False)(cls)


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
    def module(self) -> nn.Module:
        return self.NN(**dcls_asdict(self))

    @typing.override
    def children(self):
        return ()
