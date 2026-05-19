# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._types import dcls_no_repr

from .inits import MessInit

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]


@dcls_no_repr
class _BaseNorm(MessInit):
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

    @typing.override
    def _check_data(self) -> None:
        if self.num_features <= 0:
            raise ValueError(f"{self.num_features=} <= 0.")

        if self.eps <= 0:
            raise ValueError(f"{self.eps=} <= 0.")

        if self.momentum is not None and self.momentum <= 0:
            raise ValueError(f"If given, {self.momentum=} should be positive.")


class BatchNorm1d(_BaseNorm, key=nn.BatchNorm1d):
    "Applies Batch Normalization over a 2D or 3D input."


class BatchNorm2d(_BaseNorm, key=nn.BatchNorm2d):
    "Applies Batch Normalization over a 4D input."


class BatchNorm3d(_BaseNorm, key=nn.BatchNorm3d):
    "Applies Batch Normalization over a 5D input."


class InstanceNorm1d(_BaseNorm, key=nn.InstanceNorm1d):
    "Applies Instance Normalization over a 2D or 3D input."


class InstanceNorm2d(_BaseNorm, key=nn.InstanceNorm2d):
    "Applies Instance Normalization over a 4D input."


class InstanceNorm3d(_BaseNorm, key=nn.InstanceNorm3d):
    "Applies Instance Normalization over a 5D input."
