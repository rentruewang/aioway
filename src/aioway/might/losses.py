# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from aioway._common import dcls_frozen_no_repr

from .might import Might

import typing

__all__ = [
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
]

_REDUCTION = frozenset(["none", "mean", "sum"])


@dcls_frozen_no_repr
class _ReducibleLoss(Might):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented


@dcls_frozen_no_repr
class L1Loss(_ReducibleLoss):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    KEY = nn.L1Loss


@dcls_frozen_no_repr
class MSELoss(_ReducibleLoss):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y
    """

    KEY = nn.MSELoss


@dcls_frozen_no_repr
class CrossEntropyLoss(_ReducibleLoss):
    """
    This criterion computes the cross entropy loss between input logits and target.
    """

    KEY = nn.CrossEntropyLoss


@dcls_frozen_no_repr
class CTCLoss(_ReducibleLoss):
    """
    The Connectionist Temporal Classification loss.
    """

    KEY = nn.CTCLoss


@dcls_frozen_no_repr
class NLLLoss(_ReducibleLoss):
    """
    The negative log likelihood loss.
    It is useful to train a classification problem with C classes.
    """

    KEY = nn.NLLLoss


@dcls_frozen_no_repr
class KLDivLoss(_ReducibleLoss):
    """
    The Kullback-Leibler divergence loss.
    """

    KEY = nn.KLDivLoss


@dcls_frozen_no_repr
class BCELoss(_ReducibleLoss):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    """

    KEY = nn.BCELoss


@dcls_frozen_no_repr
class BCEWithLogitsLoss(_ReducibleLoss):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    """

    KEY = nn.BCEWithLogitsLoss


@dcls_frozen_no_repr
class SmoothL1Loss(_ReducibleLoss):
    """
    Creates a criterion that uses a squared term
    if the absolute element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than torch.nn.MSELoss and in some cases
    prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    """

    KEY = nn.SmoothL1Loss
