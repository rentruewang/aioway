# Copyright (c) AIoWay Authors - All Rights Reserved

"Module fwd/init modes, similar to `torch` function/dispatch modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import nn

from ..fn import TorchThunk
from ._on_off import OnOffCtx, OnOffStack

__all__ = [
    "NnFwdFn",
    "NnInitFn",
    "NnFwdMode",
    "NnInitMode",
    "module_fwd",
    "module_init",
]

LOGGER = logging.getLogger(__name__)

FORWARDS: OnOffStack[NnFwdMode] = OnOffStack()
"`NnFwdMode` that is currently entered."

INITS: OnOffStack[NnInitMode] = OnOffStack()
"`NnInitMode` that is currently entered."


def module_fwd(module: nn.Module, /, *args, **kwargs) -> typing.Any:
    """
    Call the `nn.Module`.
    `aioway` functions must call this function to call `nn.Module`.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected an `nn.Module` instance. Got {module=}.")

    return _invoke_rec(FORWARDS, NnFwdFn, module, args, kwargs)


def module_init(init: type[nn.Module], /, *args, **kwargs) -> nn.Module:
    """
    Initialize the `nn.Module`.
    `aioway` functions must call this function to initalize `nn.Module`.
    """

    if not isinstance(init, type) or not issubclass(init, nn.Module):
        raise TypeError(
            f"Your module type {init} is not valid. Must be an `nn.Module` subclass."
        )

    result = _invoke_rec(INITS, NnInitFn, init, args, kwargs)

    if not isinstance(result, nn.Module):
        raise TypeError("Function `module_init` must return an `nn.Module`.")

    return result


def _invoke_rec(
    stack: OnOffStack[typing.Any],
    fn_type: type,
    call: cabc.Callable[..., typing.Any],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    """
    Essentially, invoke the given `call` recursively until the `stack` is exhausted.

    Overriding modes must only call `module_fwd` and `module_init`,
    which in turn calls this function to pop the next `mode` off the stack, and invoke it.

    This concept is borrowed from `__torch_function__` and `__torch_dispatch__`,
    you can see similarity when reading the code around their `_pop_mode_temporarily` function,
    which corresponds to our `temp_pop` function on the stack.
    """

    if not stack:
        return call(*args, **kwargs)

    with stack.temp_pop() as mode:
        assert isinstance(mode, NnModeOnOff)

        thunk = fn_type(func=call, args=args, kwargs=kwargs)

        assert isinstance(thunk, TorchThunk)

        if mode.on:
            return mode(thunk)

        else:
            return thunk.do()


@dcls.dataclass
class NnModeOnOff[T, V = object](OnOffCtx, abc.ABC):
    """
    The mixin for either `NnFwdMode`, `NnInitMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> V:
        raise NotImplementedError

    @typing.override
    @ctxl.contextmanager
    def ctx(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        with self.STACK.hold(self):
            yield self


@dcls.dataclass
class NnFwdFn(TorchThunk[nn.Module]):
    "`NnFwdFn` represents the module calls."

    func: nn.Module
    "The module for the `Fn`."

    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def do(self) -> object:
        return module_fwd(self.func, *self.args, **self.kwargs)

    def load_state_dict(
        self,
        state_dict: cabc.Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Expose the `nn.Module.load_state_dict` function to the users.
        """

        return self.func.load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self):
        """
        Expose the `nn.Module.state_dict` function to the users.
        """

        return self.func.state_dict()


class NnFwdMode(NnModeOnOff[NnFwdFn], abc.ABC):
    """
    `NnFwdMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.ctx()` method (I prefer context managers).

    It is triggered when a `nn.Module` is called.
    """

    STACK = FORWARDS


class NnInitFn(TorchThunk[type[nn.Module]]):
    """
    `NnInitFn` is the "leftover" `nn.Module`s that are not covered by the `Might` API.
    """

    func: type[nn.Module]
    "The type of `nn.Module`."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")

    @typing.override
    def do(self) -> nn.Module:
        return module_init(self.func, *self.args, **self.kwargs)


class NnInitMode(NnModeOnOff[NnInitFn, nn.Module], abc.ABC):
    """
    `NnInitMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.ctx()` method (I prefer context managers).

    It is triggered when a `nn.Module` is initialized.
    """

    STACK = INITS
