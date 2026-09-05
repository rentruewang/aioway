# Copyright (c) AIoWay Authors - All Rights Reserved

"The `LossTSpec` interface."

import torch
from torchrl.data import tensor_specs as tspecs

from aioway.schemas import Device, DeviceLike, DType, DTypeLike

__all__ = ["LossTSpec"]


class LossTSpec(tspecs.Unbounded):
    """
    The `TSpec` that will be marked as losses.
    """

    def __init__(
        self, device: DeviceLike | None = None, dtype: DTypeLike | None = None
    ) -> None:
        super().__init__(
            shape=torch.Size(()),
            device=Device.parse(device).torch() if device is not None else None,
            dtype=DType.parse(dtype).torch() if dtype is not None else None,
        )

        if self.shape:
            raise ValueError(f"Only empty shape is allowed. {self.shape=}.")
