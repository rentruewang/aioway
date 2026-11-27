# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

from torchvision import datasets as tvdsets
from torchvision.transforms import v2 as tvt

from aioway.cmds.nets import (
    FloatRangeListDomain,
    IntRangeItemDomain,
    float_range_list,
    int_range_item,
)
from aioway.io import InputTarget, InputTargetLikeDset

__all__ = ["MnistDataset"]


class MnistDataset(InputTargetLikeDset):
    def __init__(self) -> None:
        self._mnist = tvdsets.MNIST(pathlib.Path.home(), download=True)

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, idx) -> InputTarget:
        x, y = self._mnist[idx]
        x = tvt.ToTensor()(x)
        return InputTarget(x, y)

    @typing.no_type_check
    def __getitems__(self, idx):
        return td.stack([self[i] for i in idx])

    @property
    def input_domain(self) -> FloatRangeListDomain:
        "`ToTensor` gives `[1, 28, 28]` in `[0, 1]`."

        return float_range_list(0, 1, 1, 28, 28)

    @property
    def target_domain(self) -> IntRangeItemDomain:
        return int_range_item(0, 9)

    @property
    def collate_fn(self):
        return lambda x: x
