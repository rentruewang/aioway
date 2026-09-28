# Copyright (c) AIoWay Authors - All Rights Reserved

"Layout describes how the tensor is stored in memory."

import typing

import torch

from ._bases import TorchAttrBase

__all__ = ["Layout", "LayoutLike"]

type LayoutLike = str | torch.layout | Layout


class Layout(TorchAttrBase[torch.layout]):
    """
    The class corresponding to `torch.layout` class.
    """

    TYPE = torch.layout

    def __str__(self):
        return str(self._data).removeprefix("torch.")

    @typing.override
    def __getstate__(self) -> object:
        return str(self)

    @typing.override
    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def parse(cls, layout: LayoutLike):
        if isinstance(layout, Layout):
            return layout

        if isinstance(layout, str):
            return cls.parse(getattr(torch, layout))

        if isinstance(layout, torch.layout):
            return cls(layout)

        raise ValueError(f"Cannot parse {layout=}.")
