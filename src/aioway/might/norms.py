# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._common import dcls_frozen_no_repr

from .might import Might

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]


@dcls_frozen_no_repr
class _BaseNorm(Might):
    "Base normalization layer for shared code of batch norm and instance norm."

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

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

    affine: bool = True
    """
    A boolean value that when set to `True`, this module has learnable affine parameters.
    Default: `True`.
    """

    track_running_stats: bool = True
    """
    A boolean value that when set to `True`, this module tracks the running mean and variance,
    and when set to `False`, this module does not track such statistics,
    and initializes statistics buffers running_mean and running_var as None.
    When these buffers are None, this module always uses batch statistics.
    In both training and eval modes. Default: `True`
    """

    bias: bool = True
    """
    If set to `False`, the layer will not learn an additive bias,
    only relevant if affine is `True`. Default: `True`
    """

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError(f"{self.num_features=} <= 0.")

        if self.eps <= 0:
            raise ValueError(f"{self.eps=} <= 0.")

        if self.momentum is not None and self.momentum <= 0:
            raise ValueError(f"If given, {self.momentum=} should be positive.")


class BatchNorm1d(_BaseNorm):
    "Applies Batch Normalization over a 2D or 3D input."

    KEY = nn.BatchNorm1d


class BatchNorm2d(_BaseNorm):
    "Applies Batch Normalization over a 4D input."

    KEY = nn.BatchNorm2d


class BatchNorm3d(_BaseNorm):
    "Applies Batch Normalization over a 5D input."

    KEY = nn.BatchNorm3d


class InstanceNorm1d(_BaseNorm):
    "Applies Instance Normalization over a 2D or 3D input."

    KEY = nn.InstanceNorm1d


class InstanceNorm2d(_BaseNorm):
    "Applies Instance Normalization over a 4D input."

    KEY = nn.InstanceNorm2d


class InstanceNorm3d(_BaseNorm):
    "Applies Instance Normalization over a 5D input."

    KEY = nn.InstanceNorm3d
