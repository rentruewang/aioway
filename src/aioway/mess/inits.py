# Copyright (c) AIoWay Authors - All Rights Reserved

from .mess import MessInit, mess_init_dcls

__all__ = ["mess_init_dcls", "MessInit", "LossInit", "NormInit"]


@mess_init_dcls
class LossInit(MessInit):
    "Base layer for `Loss` layers, which does not accept arguments."


@mess_init_dcls
class NormInit(MessInit):
    "Base normalization layer for shared code of batch norm and instance norm."

    num_features: int
    "The number of features C of the output."

    eps: float = 1e-5
    "A value added to the denominator for numerical stability. Default: 1e-5"

    momentum: float | None = 0.1
    """
    the value used for the running_mean and running_var computation.
    Can be set to None for cumulative moving average (i.e. simple average).
    Default: 0.1.
    """

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError(f"{self.num_features=} <= 0.")

        if self.eps <= 0:
            raise ValueError(f"{self.eps=} <= 0.")

        if self.momentum is not None and self.momentum <= 0:
            raise ValueError(f"If given, {self.momentum=} should be positive.")
