# Copyright (c) AIoWay Authors - All Rights Reserved


import dataclasses as dcls

from torch import nn

from .fwds import MessFwd
from .inits import MessInit

__all__ = ["Mess"]

_MESS_REGISTRY: dict[type[nn.Module], Mess] = {}


@dcls.dataclass(frozen=True, slots=True)
class Mess:
    """
    `Mess` is the runtime information for `nn.Module`,
    containing information of `nn.Module.forward`.

    `Mess` stands for [m]odule [e]xecution [s]ignature [s]ystem.

    It carries both a `MessInit` type and a `MessFwd` type,
    for the signatures of initialization time / run time respectively.
    """

    nn_type: type[nn.Module]
    """
    The `nn.Module` type that has the init signature `init` and forward signature `fwd`.
    """

    init: type[MessInit]
    """
    The initialization signature of the `nn.Module`.
    """

    fwd: type[MessFwd]
    """
    The runtime signature of the `nn.Module`.
    """

    def __post_init__(self) -> None:
        # Log `self` in the registry.
        _MESS_REGISTRY[self.nn_type] = self


def find_mess(nn_type: type[nn.Module]) -> Mess:
    """
    Find the `Mess` instance that is previously initialized.
    It carries information on both `MessInit` and `MessFwd`,
    and their corresponding `nn_type`.

    If the `Mess` is not found, `NotImplemented` is returned.
    """

    return _MESS_REGISTRY.get(nn_type, NotImplemented)
