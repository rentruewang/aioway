# Copyright (c) AIoWay Authors - All Rights Reserved

import frozenlist
from aioway._utils import Sign
from torch import nn
import typing
from collections import abc as cabc
import dataclasses as dcls

__all__ = ["MatchFunc", "Matcher"]


class Matcher:
    """
    The function for matching.
    """

    def __init__(self) -> None:
        self._registry: dict[type[nn.Module], FuncType] = {}

    def __call__(self, module: nn.Module, /) -> nn.Module:
        for nn_type, function in self._registry.items():
            if not isinstance(module, nn_type):
                continue

            return function(module)

        return NotImplemented

    def register[M: MatchFunc](self, func: M) -> M:
        sign = Sign.from_callable(func)
        param_type = _match_func_param_type(sign)
        return_type = _match_func_return_type(sign)
        self._registry[param_type] = FuncType(func, return_type)
        return func

    def keys(self) -> cabc.KeysView[type[nn.Module]]:
        return self._registry.keys()


class MatchFunc[M: nn.Module = typing.Any, N: nn.Module = typing.Any](typing.Protocol):
    "The function in charge of processing."

    def __call__(self, module: M, /) -> N: ...


@dcls.dataclass(frozen=True)
class FuncType[I: nn.Module = typing.Any, O: nn.Module = typing.Any]:
    func: MatchFunc[I, O]
    ret_type: type[nn.Module]

    def __call__(self, module: I) -> O:
        result = self.func(module)
        if not isinstance(result, self.ret_type):
            raise TypeError(f"The return type {type(result)=} is not {self.ret_type=}.")
        return result


def _match_func_param_type(sign: Sign) -> type[nn.Module]:
    [param] = sign.param_list
    annotation = param.annotation

    if param.is_any_type:
        return NotImplemented

    if not issubclass(annotation, nn.Module):
        raise TypeError(f"{annotation=} is not subclass of `nn.Module`.")

    return annotation


def _match_func_return_type(sign: Sign) -> type[nn.Module]:
    if sign.returns_any_type:
        return NotImplemented

    return_type = sign.return_annotation

    if not issubclass(return_type, nn.Module):
        raise TypeError(f"{return_type=} is not subclass of `nn.Module`.")

    return return_type
