# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified registry for `type[nn.Module]` storing `aioway` operations."

import dataclasses as dcls

from torch import nn

from aioway._utils import Sign, is_nn_type

from .signs import sign_reg


@dcls.dataclass
class NnOp:
    """
    The dataclass that stores all the operations associated `type[nn.Module]`.

    This is a shared location s.t. we can store all needed operations together,
    without having to import multiple functions to access multiple utilities.
    """

    sign: Sign | None = None
    "The signature of `nn.Module` using `.forward`."

    def __bool__(self) -> bool:
        """
        Check if `NnOp` has any attribute defined.
        """

        return any(val for val in dcls.asdict(self).values())


def nn_op(module: type[nn.Module]) -> NnOp:
    """
    Get all the operations associated with a specific type.
    """

    if not is_nn_type(module):
        raise TypeError(type(module))

    return NnOp(sign=sign_reg()[module])
