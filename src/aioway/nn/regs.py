# Copyright (c) AIoWay Authors - All Rights Reserved

"A unified registry for `type[nn.Module]` storing `aioway` operations."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import Sign

if typing.TYPE_CHECKING:
    from .deductions import Deduction

__all__ = ["NnOp", "NnReg", "nn_reg", "with_nn_reg", "NnRegAttr"]


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
        return self.deduction is not None


class NnReg(cabc.MutableMapping[type[nn.Module], NnOp]):
    """
    The registry for `type[nn.Module]`.
    """

    def __init__(self, reg: cabc.Mapping[type[nn.Module], NnOp] | None = None) -> None:
        reg = reg or {}
        self._reg = reg if isinstance(reg, dict) else dict(reg)

    def __repr__(self) -> str:
        return repr(self._reg)

    def __len__(self) -> int:
        return len(self._reg)

    def __iter__(self) -> cabc.Iterator[type[nn.Module]]:
        yield from self._reg

    def __getitem__(self, key: type[nn.Module]) -> NnOp:
        return self._reg[key]

    def __setitem__(self, key: type[nn.Module], op: NnOp):
        self._reg[key] = op

    def __delitem__(self, key: type[nn.Module]):
        del self._reg[key]

    def insert_if_empty(self, key: type[nn.Module]) -> None:
        "Insert the key if nothing is defined on it."
        if key not in self:
            self[key] = NnOp()

    def delete_if_empty(self, key: type[nn.Module]) -> None:
        "Delete the key if nothing is defined on it."
        if key not in self:
            return

        if not self[key]:
            del self[key]


_nn_reg: NnReg = NnReg()


def nn_reg() -> NnReg:
    """
    The current registry for `type[nn.Module]`s.
    """

    return _nn_reg


@ctxl.contextmanager
def with_nn_reg(
    registry: cabc.Mapping[type[nn.Module], NnOp] | None = None,
) -> cabc.Generator[NnReg]:
    """
    Overwrite the current registry and restore later.

    Yield the input for convenience in inline usage.
    """

    global _nn_reg

    _nn_reg, before = NnReg(registry), _nn_reg

    try:
        yield _nn_reg
    finally:
        _nn_reg = before


class NnRegAttr[T](cabc.MutableMapping[type[nn.Module], T]):
    """
    `NnRegAttr` is a view on `NnReg`'s stored attributes.

    For each getitem / setitem / delitem / contains operation,
    it checks if `NnReg` has the `NnOp`, and whether the `NnOp` has the attr.
    """

    def __init__(self, attr_key: str, val_type: type[T], nn_reg: NnReg):
        self._nn_reg = nn_reg
        self._attr_key = attr_key
        self._val_type = val_type

    def __repr__(self) -> str:
        return f"DeductionRegistry({len(self)})"

    def __len__(self) -> int:
        return len(self._nn_reg)

    def __contains__(self, key) -> bool:
        if key not in self._nn_reg:
            return False

        if getattr(self._nn_reg[key], self._attr_key) is None:
            return False

        return True

    def __getitem__(self, key: type[nn.Module]) -> T:
        nn_op = self._nn_reg[key]

        if (attr := getattr(nn_op, self._attr_key, None)) is None:
            raise KeyError(key)

        assert isinstance(attr, self._val_type)
        return attr

    def __setitem__(self, key: type[nn.Module], val: T):
        if not isinstance(val, self._val_type):
            raise TypeError(f"{val} should be of type {self._val_type}.")

        reg = self._nn_reg

        reg.insert_if_empty(key)
        nn_op = reg[key]

        setattr(nn_op, self._attr_key, val)

    def __delitem__(self, key: type[nn.Module]) -> None:
        reg = self._nn_reg

        nn_op = reg[key]
        setattr(nn_op, self._attr_key, None)

        reg.delete_if_empty(key)

    def __iter__(self) -> cabc.Iterator[type[nn.Module]]:
        yield from nn_reg()
