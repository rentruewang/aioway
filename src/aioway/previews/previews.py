# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch import nn

from aioway._common import dcls_frozen_no_repr
from aioway.op import Op

__all__ = ["Preview", "find_preview", "all_previews"]


@dcls_frozen_no_repr
class Preview(Op[type[nn.Module]], abc.ABC):
    """
    `Preview` is a preview of how an `nn.Module` would be initialized.

    It provides metadata as to what `nn.Module` arguments are valid or not,
    much like how `Fate`'s objects mimicks the function signature of `torch.ops.aten.*`.

    Even though the name `Preview` sounds quite generic and perhaps confusing,
    the term is coined even before `aioway` (I think for a month),
    so for historical reasons, I won't touch it.
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    @classmethod
    @typing.override
    def name(cls) -> str:
        return "preview::" + cls.__name__


def find_preview(nn_type: type[nn.Module], *args, **kwargs) -> Preview:
    """
    Get a `Preview` from the `nn.Module` type. If not found, return `NotImplemented`.
    """

    # Right now, each `Preview` should have distinct key, so just return the 1st.
    # Just get the type. If an error is raised, construction failed,
    # pass the error back, since upper level signature failed.
    for preview_type in Preview.find(nn_type):
        return preview_type(*args, **kwargs)

    # No implementation found.
    return NotImplemented


@typing.no_type_check
def all_previews():
    """
    Get the registry for previews.
    """

    return list(_iter_previews(Preview))


def _iter_previews(cls: type[Preview]) -> cabc.Generator[type[Preview]]:
    for sub in cls.__subclasses__():
        if sub.is_concrete():
            yield sub

        yield from _iter_previews(sub)
