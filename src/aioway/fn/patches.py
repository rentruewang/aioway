# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import torch
from torch import _ops

from aioway.ctx import enabled_fake_mode

__all__ = ["register_fake_mode_patch"]

PATCHES: dict[_ops.OpOverload, cabc.Callable[..., torch.Tensor]] = {}


def find_patch(op: _ops.OpOverload) -> cabc.Callable[..., torch.Tensor]:
    """
    If a patch is found, return it. Else return `NotImplemented`.
    """

    if op in PATCHES:
        return PATCHES[op]

    return NotImplemented


def register_fake_mode_patch(op: _ops.OpOverload):
    """
    Register a patching function that only runs under fake mode.
    """

    def decorator[**P, T: torch.Tensor](f: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
        def function(*args: P.args, **kwargs: P.kwargs) -> T:
            if not enabled_fake_mode():
                raise RuntimeError("This function only runs under fake mode!")

            return f(*args, **kwargs)

        if (prev := PATCHES.get(op)) is not None:
            raise KeyError(f"You already registered a patch {prev} for {op=}.")

        PATCHES[op] = function
        return function

    return decorator
