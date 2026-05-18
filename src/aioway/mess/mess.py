# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from torch import nn

from aioway._keyed import Keyed
from aioway._types import dcls_no_repr
from aioway.mess.fwds import MessFwd
from aioway.mess.inits.inits import find_mess_init

from .inits import MessInit

__all__ = ["Mess"]


@dcls_no_repr
class Mess(abc.ABC):
    """
    `Mess` is the runtime information for `nn.Module`,
    containing information of `nn.Module.forward`.

    `Mess` stands for [m]odule [e]xecution [s]ignature [s]ystem.

    It carries both a `MessInit` type and a `MessFwd` type,
    for the signatures of initialization time / run time respectively.
    """

    init: type[MessInit]
    """
    The initialization signature of the `nn.Module`.
    """

    fwd: type[MessFwd]
    """
    The runtime signature of the `nn.Module`.
    """

    @classmethod
    def find(cls, nn_type: type[nn.Module]) -> typing.Self:

        # Right now, each `MessInit` should have distinct key, so just return the 1st.
        # Just get the type. If an error is raised, construction failed,
        # pass the error back, since upper level signature failed.
        for mess_type in MessInit.find(nn_type):
            return mess_type
        else:
            return NotImplemented
