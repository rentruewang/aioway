# Copyright (c) AIoWay Authors - All Rights Reserved

import logging
import typing

import torch

from ._bases import TorchAttrBase

__all__ = ["Device", "DeviceLike"]

LOGGER = logging.getLogger(__name__)


type DeviceLike = str | torch.device | Device
"Types convertible to a `Device`."


class Device(TorchAttrBase[torch.device]):
    """
    The device that the tensor data resides on (and will be used for compute).
    """

    __match_args__ = ("device",)
    TYPE = torch.device

    @typing.override
    def __getstate__(self) -> object:
        return str(self._data)

    @typing.override
    def __hash__(self) -> int:
        return hash(self.device)

    @typing.override
    def __str__(self) -> str:
        return repr(self._data)

    @property
    def device(self):
        return self._data

    @classmethod
    def parse(cls, device: DeviceLike) -> typing.Self:
        "The convenient wrapper to create a `Device` from compatible types."

        if isinstance(device, cls):
            return device

        if isinstance(device, str):
            return cls.parse(torch.device(device))

        if isinstance(device, torch.device):
            return cls(device)

        raise ValueError(f"Parsing failed for {device=}.")
