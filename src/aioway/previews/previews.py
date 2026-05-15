# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import inspect
import typing

from torch import nn

__all__ = ["Preview", "find_preview", "all_previews"]

_PREVIEWS_REGISTRY: dict[type[nn.Module], type[Preview]] = {}


class Preview(abc.ABC):
    """
    `Preview` is a preview of how an `nn.Module` would be initialized.

    It provides metadata as to what `nn.Module` arguments are valid or not,
    much like how `Fate`'s objects mimicks the function signature of `torch.ops.aten.*`.

    Even though the name `Preview` sounds quite generic and perhaps confusing,
    the term is coined even before `aioway` (I think for a month),
    so for historical reasons, I won't touch it.
    """

    NN: typing.ClassVar[type[nn.Module]]
    "The `nn.Module` type that should be implemented."

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @abc.abstractmethod
    def do(self) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def _register(cls) -> None:
        # Skip abstract class and don't register it.
        if inspect.isabstract(cls):
            return

        # Abstract class that does not define `ClassVar` is not properly captured by `inspect.isabstract`.
        try:
            nn_type = cls.NN
        except AttributeError:
            return

        if not isinstance(nn_type, type) or not issubclass(nn_type, nn.Module):
            raise TypeError(f"{cls}.NN={nn_type} is not an `nn.Module`.")

        if prev := _PREVIEWS_REGISTRY.get(cls.NN):
            raise KeyError(f"Another implementation for {cls.NN} found: {prev}.")

        _PREVIEWS_REGISTRY[cls.NN] = cls


def find_preview(nn_type: type[nn.Module], *args, **kwargs) -> Preview:
    """
    Get a `Preview` from the `nn.Module` type. If not found, return `NotImplemented`.
    """

    if (preview_type := _PREVIEWS_REGISTRY.get(nn_type, None)) is None:
        return NotImplemented

    # Just get the type. If an error is raised, construction failed,
    # pass the error back, since upper level signature failed.
    return preview_type(*args, **kwargs)


def all_previews():
    """
    Get the registry for previews.
    """

    return _PREVIEWS_REGISTRY
