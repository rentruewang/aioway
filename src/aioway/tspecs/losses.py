# Copyright (c) AIoWay Authors - All Rights Reserved

"The `LossTSpec` interface."

from torchrl.data import tensor_specs as tspecs

__all__ = ["LossTSpec"]


class LossTSpec(tspecs.Unbounded):
    """
    The `TSpec` that will be marked as losses.
    """

    def __post_init__(self) -> None:
        if self.shape:
            raise ValueError(f"Only empty shape is allowed. {self.shape=}.")
