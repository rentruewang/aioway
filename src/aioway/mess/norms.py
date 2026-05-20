# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


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


_ = Mess(nn_type=nn.BatchNorm1d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 2D or 3D input."


_ = Mess(nn_type=nn.BatchNorm2d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 4D input."


_ = Mess(nn_type=nn.BatchNorm3d, init=NormInit, fwd=InputFwd)
"Applies Batch Normalization over a 5D input."


_ = Mess(nn_type=nn.InstanceNorm1d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 2D or 3D input."


_ = Mess(nn_type=nn.InstanceNorm2d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 4D input."


_ = Mess(nn_type=nn.InstanceNorm3d, init=NormInit, fwd=InputFwd)
"Applies Instance Normalization over a 5D input."
