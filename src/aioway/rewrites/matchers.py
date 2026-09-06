# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import Sign

__all__ = ["MatchFunc", "Matcher"]


@dcls.dataclass(frozen=True)
class FuncOutType[I: nn.Module = typing.Any, O: nn.Module = typing.Any]:
    func: MatchFunc[I, O]
    ret_type: type[nn.Module]

    def __call__(self, module: I) -> O:
        result = self.func(module)
        if not isinstance(result, self.ret_type):
            raise TypeError(f"The return type {type(result)=} is not {self.ret_type=}.")
        return result


@dcls.dataclass(frozen=True)
class FuncInOutType[I: nn.Module = typing.Any, O: nn.Module = typing.Any](FuncOutType):
    arg_type: type[nn.Module]

    def __repr__(self) -> str:
        def type_name(typ: type):
            return typ.__module__ + "." + typ.__name__

        return f"({type_name(self.arg_type)}) -> {type_name(self.ret_type)}"


class Matcher(cabc.Mapping[type[nn.Module], FuncInOutType]):
    """
    The function for matching.
    """

    def __init__(self) -> None:
        self._registry: dict[type[nn.Module], FuncOutType] = {}

    def __call__(self, module: nn.Module, /) -> nn.Module:
        for nn_type, function in self._registry.items():
            if not isinstance(module, nn_type):
                continue

            return function(module)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._registry)

    def __getitem__(self, key: type[nn.Module]) -> FuncInOutType:
        func_type = self._registry[key]
        return FuncInOutType(
            func=func_type.func, ret_type=func_type.ret_type, arg_type=key
        )

    def __iter__(self) -> cabc.Iterator[type[nn.Module]]:
        yield from self.keys()

    def register[M: MatchFunc](self, func: M) -> M:
        sign = Sign.from_callable(func)
        param_type = _match_func_param_type(sign)
        return_type = _match_func_return_type(sign)
        self._registry[param_type] = FuncOutType(func, return_type)
        return func

    def keys(self) -> cabc.KeysView[type[nn.Module]]:
        return self._registry.keys()


class MatchFunc[M: nn.Module = typing.Any, N: nn.Module = typing.Any](typing.Protocol):
    "The function in charge of processing."

    def __call__(self, module: M, /) -> N: ...


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
