# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Explainer` interface."

import inspect
import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from captum import attr
from torch import nn

from aioway._utils import dcls_asdict, is_seq_of

__all__ = ["Expl", "expl_dcls"]

_EXPLAINER_TYPES: dict[str, type[Expl]] = {}
"The explainer corresponding to its unique identifiers."


@typing.dataclass_transform(frozen_default=True)
def expl_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@expl_dcls
class Expl(abc.ABC):
    """
    The explainer based on `captum.attr.Attribution`.

    Each instance must be able to be converted to the runtime keyword arguments.
    """

    _CAPTUM_CLASS: typing.ClassVar[type[attr.Attribution]] = attr.Attribution
    "The corresponding captum class. If not overwritten, the class is abstract."

    def __init_subclass__(cls) -> None:
        if _is_abstract_explainer(cls):
            return

        # Using the `captum.attr.*` name as the name of the class.
        name = cls._CAPTUM_CLASS.__name__

        if name in _EXPLAINER_TYPES:
            existing = _EXPLAINER_TYPES[name]
            raise KeyError(
                f"{name=} is already registered by {existing}. "
                f"But {cls} also has the same name."
            )

        _EXPLAINER_TYPES[name] = cls

    def __call__(
        self, module: nn.Module, *inputs: torch.Tensor
    ) -> cabc.Iterator[torch.Tensor]:
        """
        This function corresponds to `captum.attr.*`'s `.attribute` method.

        Since `captum.attr.*.attribute` may return a `torch.Tensor` or a sequence of it,
        to match the module's input, this yields an `Iterator[torch.Tensor]`,
        s.t. they can share the same API module(*iterator of tensor).

        Args:
            module: An `nn.Module`, must output a scalar tenosr.
            inputs: Inputs that will be fed to the `module`.
        """

        captum_attr = self._CAPTUM_CLASS(module)
        result = captum_attr.attribute(inputs, **self._kwargs())

        if isinstance(result, torch.Tensor):
            yield result
            return

        if is_seq_of(torch.Tensor)(result):
            yield from result
            return

        raise TypeError(f"{result=} from `captum.attr` has unexpected type.")

    def _kwargs(self) -> dict[str, typing.Any]:
        """
        The kwargs to feed to the explainer's runtime.
        The signature of this class can change anytime, as support broadens.
        """

        return dcls_asdict(self)


def _is_abstract_explainer(cls: type) -> bool:
    if not issubclass(cls, Expl):
        raise TypeError

    if inspect.isabstract(cls):
        return True

    if cls._CAPTUM_CLASS is attr.Attribution:
        return True

    return False
