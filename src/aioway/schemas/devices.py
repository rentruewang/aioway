# Copyright (c) AIoWay Authors - All Rights Reserved

import logging
import typing

import torch

from aioway._common import get_tracker

__all__ = ["Device", "DeviceLike"]

LOGGER = logging.getLogger(__name__)
TRACKER = get_tracker(lambda: Device)


type DeviceLike = str | torch.device | Device
"Types convertible to a `Device`."


class Device:
    """
    The device that the tensor data resides on (and will be used for compute).
    """

    __match_args__ = ("device",)

    def __init__(self, device: str | torch.device) -> None:
        # On top of only needing to store `torch.device`,
        # it also does a check to ensure that if a string is passed,
        # the device is valid, and must follow the "device[:index]" format.
        try:
            self._device = torch.device(device)
        except RuntimeError as e:
            raise ValueError("Not a valid `torch.device`.") from e

        LOGGER.debug("Device %s instance created", self)

    @typing.override
    def __getstate__(self) -> object:
        return str(self._device)

    @typing.override
    def __hash__(self) -> int:
        return hash(self.device)

    @typing.override
    def __eq__(self, other: object) -> bool:
        match other:
            case torch.device():
                return self._device == other
            case Device(device):
                return self._device == device
            case str():
                # Instead of converting `other` with `torch.device`,
                # which may fail, compare the string directly.
                return str(self._device) == other

        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        return str(self)

    @typing.override
    def __str__(self) -> str:
        return str(self._device)

    @property
    def device(self):
        return self._device

    def torch(self):
        """
        Convert to `torch.device`.
        """

        return self._device

    @classmethod
    def parse(cls, device: DeviceLike) -> Device:
        "The convenient wrapper to create a `Device` from compatible types."

        if isinstance(device, cls):
            return device

        if isinstance(device, str | torch.device):
            return Device(device)

        raise TypeError(f"Cannot handle {type(device)=}.")
