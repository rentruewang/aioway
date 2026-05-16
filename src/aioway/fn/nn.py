# Copyright (c) AIoWay Authors - All Rights Reserved

"Module fwd/init modes, similar to `torch` function/dispatch modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from .fn import OnOffCtx, OnOffStack, TorchThunk

__all__ = ["NnFwdFn", "NnInitFn"]

_FORWARDS: OnOffStack[MFwdMode] = OnOffStack()
"`MFwdMode` that is currently entered."

_INITS: OnOffStack[MInitMode] = OnOffStack()
"`MInitMode` that is currently entered."


@ctxl.contextmanager
def set_nn_mode(fwd: bool, init: bool):
    """
    Turn on or off the modes for

    Args:
        fwd: Disable the `MFwdMode` mode if `True`.
        init: Disable the `MInitMode` mode if `True`.

    Note:
        This is similar to `set_torch_mode`.
    """

    with _FORWARDS.switch(fwd), _INITS.switch(init):
        yield


@dcls.dataclass
class MModeOnOff[T](OnOffCtx, abc.ABC):
    """
    The mixin for either `MFwdMode`, `MInitMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> object:
        raise NotImplementedError

    @typing.override
    @ctxl.contextmanager
    def ctx(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        if not self.on:
            yield self
            return

        with self.STACK.enter(self):
            yield self


@dcls.dataclass
class NnFwdFn(TorchThunk[nn.Module]):
    "`NnFwdFn` represents the module calls."

    func: nn.Module
    "The module for the `Fn`."

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


class MFwdMode(MModeOnOff[NnFwdFn], abc.ABC):
    """
    `MFwdMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.ctx()` method (I prefer context managers).

    It is triggered when a `nn.Module` is called.
    """

    STACK = _FORWARDS


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


class MInitMode(MModeOnOff[NnInitFn], abc.ABC):
    """
    `MInitMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.ctx()` method (I prefer context managers).

    It is triggered when a `nn.Module` is initialized.
    """

    STACK = _INITS
