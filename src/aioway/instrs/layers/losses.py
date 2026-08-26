# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from ..nn import NnInstr, nn_instr_dcls

__all__ = [
    "BaseLoss",
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


@nn_instr_dcls
class BaseLoss(NnInstr):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """


@nn_instr_dcls
class L1Loss(BaseLoss):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    NN = nn.L1Loss


@nn_instr_dcls
class MSELoss(BaseLoss):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y
    """

    NN = nn.MSELoss


@nn_instr_dcls
class CrossEntropyLoss(BaseLoss):
    """
    This criterion computes the cross entropy loss between input logits and target.
    """

    NN = nn.CrossEntropyLoss


@nn_instr_dcls
class CTCLoss(BaseLoss):
    """
    The Connectionist Temporal Classification loss.
    """

    NN = nn.CTCLoss


@nn_instr_dcls
class NLLLoss(BaseLoss):
    """
    The negative log likelihood loss.
    It is useful to train a classification problem with C classes.
    """

    NN = nn.NLLLoss


@nn_instr_dcls
class KLDivLoss(BaseLoss):
    """
    The Kullback-Leibler divergence loss.
    """

    NN = nn.KLDivLoss


@nn_instr_dcls
class BCELoss(BaseLoss):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    """

    NN = nn.BCELoss


@nn_instr_dcls
class BCEWithLogitsLoss(BaseLoss):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    """

    NN = nn.BCEWithLogitsLoss


@nn_instr_dcls
class SmoothL1Loss(BaseLoss):
    """
    Creates a criterion that uses a squared term
    if the absolute element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than torch.nn.MSELoss and in some cases
    prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    """

    NN = nn.SmoothL1Loss
