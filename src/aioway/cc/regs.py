# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified registry for `type[nn.Module]` storing `aioway` operations."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import Sign, dcls_asdict

if typing.TYPE_CHECKING:
    from .deductions import Deduction

__all__ = ["NnOp", "NnRegView", "nn_reg", "deduction_reg", "sign_reg"]

type _NnModuleReg[T] = dict[type[nn.Module], T]

_SIGN_REG: _NnModuleReg[Sign] = {}
"The signature registry for `nn.Module.forward`."

_DEDUCTION_REG: _NnModuleReg[Deduction] = {}
"The deduction function for each `nn.Module.forward` call."


@dcls.dataclass
class NnOp:
    """
    The dataclass that stores all the operations associated `type[nn.Module]`.

    This is a shared location s.t. we can store all needed operations together,
    without having to import multiple functions to access multiple utilities.
    """

    signature: Sign | None = None
    "The signature of `nn.Module` using `.forward`."

    deduction: Deduction | None = None
    "The deduction function."

    def __bool__(self) -> bool:
        """
        Check if `NnOp` has any attribute defined.
        """

        return any(val for val in dcls_asdict(self).values())


def deduction_reg() -> dict[type[nn.Module], Deduction]:
    "The deduction register."

    return _DEDUCTION_REG


def sign_reg() -> dict[type[nn.Module], Sign]:
    "The signature register."

    return _SIGN_REG


def nn_reg() -> NnRegView:
    return NnRegView()


class NnRegView(cabc.Mapping[type[nn.Module], NnOp]):
    """
    The registry for `type[nn.Module]`.

    Note that this is a view type, therefore does not support assignment,
    and that the registeries it stores is frozen since initialization,
    and mutation to the registries during the lifetime will not affect it.

    The operations on this registry view type is cached.
    """

    def __init__(self) -> None:
        self._deductions = {**_DEDUCTION_REG}
        self._signs = {**_SIGN_REG}

    def __repr__(self) -> str:
        return repr({key: self[key] for key in self._keys})

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, module: object) -> bool:
        return module in self._keys

    def __iter__(self) -> cabc.Iterator[type[nn.Module]]:
        yield from self._keys

    def __getitem__(self, key: type[nn.Module]) -> NnOp:
        return self.__getitem(key)

    @functools.cache
    def __getitem(self, key: type[nn.Module]) -> NnOp:
        if key not in self._keys:
            raise KeyError(key)

        deduction = self._deductions.get(key)
        sign = self._signs.get(key)

        return NnOp(deduction=deduction, signature=sign)

    @functools.cached_property
    def _keys(self) -> set[type[nn.Module]]:
        return _union_of_keys(self._deductions, self._signs)


def _union_of_keys(*dicts: _NnModuleReg) -> set[type[nn.Module]]:
    keys: set[type[nn.Module]] = set()

    for reg in dicts:
        keys |= reg.keys()

    return keys
