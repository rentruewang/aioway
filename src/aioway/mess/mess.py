# Copyright (c) AIoWay Authors - All Rights Reserved


import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway.renders import camel_to_snake, render_fcall

__all__ = ["Mess", "find_mess", "list_mess"]

_MESS_REGISTRY: dict[type[nn.Module], Mess] = {}


@typing.dataclass_transform(frozen_default=True)
def mess_init_dcls(cls):
    "Decorator of dataclass for `MessInit`."
    return dcls.dataclass(frozen=True, repr=False)(cls)


@mess_init_dcls
class MessInit(abc.ABC):
    """
    `MessInit` is a preview of how an `nn.Module` would be initialized.
    It is the init part of `Mess`.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    def __repr__(self) -> str:
        return render_fcall(
            "mess_init::" + camel_to_snake(type(self).__name__), **dcls.asdict(self)
        )


@typing.dataclass_transform(frozen_default=True)
def mess_fwd_dcls(cls):
    "Decorator of dataclass for `MessInit`."
    return dcls.dataclass(frozen=True, repr=False)(cls)


@mess_fwd_dcls
class MessFwd(abc.ABC):
    """
    `MessFwd` contains info about how a module's runtime signature looks like.
    """

    def __repr__(self) -> str:
        return render_fcall(
            "mess_fwd::" + camel_to_snake(type(self).__name__), **dcls.asdict(self)
        )


@typing.final
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


def list_mess():
    return _MESS_REGISTRY
