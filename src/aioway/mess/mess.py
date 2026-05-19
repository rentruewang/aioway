# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._types import dcls_no_repr

from .fwds import MessFwd
from .inits import MessInit

__all__ = ["Mess"]


@dcls_no_repr
class Mess:
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
        """
        Perform a search for both `MessInit` and `MessFwd` on `nn_type`.
        If either returns `NotImplemented`, `NotImplemented` is returned.
        Or else return a `Mess` which contains both types.
        """

        # Right now, both `MessInit` / `MessFwd` should have distinct key,
        # so just return the 1st.

        for init_type in MessInit.find(nn_type):
            break
        else:
            return NotImplemented

        for fwd_type in MessFwd.find(nn_type):
            break
        else:
            return NotImplemented

        return cls(init=init_type, fwd=fwd_type)
