# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from torch import nn

__all__ = ["module_init", "module_forward"]


def module_init[**P, T: nn.Module](
    _mod: cabc.Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    return _mod(*args, **kwargs)


def module_forward(
    _mod: cabc.Callable[..., typing.Any], /, *args, **kwargs
) -> typing.Any:
    return _mod(*args, **kwargs)
