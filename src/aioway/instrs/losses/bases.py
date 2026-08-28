# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from ..nn import BaseLossInstr, instr_dcls

__all__ = [
    "BaseLossInstr",
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


@instr_dcls
class L1Loss(BaseLossInstr):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    NN = nn.L1Loss


@instr_dcls
class MSELoss(BaseLossInstr):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y
    """

    NN = nn.MSELoss


@instr_dcls
class CrossEntropyLoss(BaseLossInstr):
    """
    This criterion computes the cross entropy loss between input logits and target.
    """

    NN = nn.CrossEntropyLoss


@instr_dcls
class CTCLoss(BaseLossInstr):
    """
    The Connectionist Temporal Classification loss.
    """

    NN = nn.CTCLoss


@instr_dcls
class NLLLoss(BaseLossInstr):
    """
    The negative log likelihood loss.
    It is useful to train a classification problem with C classes.
    """

    NN = nn.NLLLoss


@instr_dcls
class KLDivLoss(BaseLossInstr):
    """
    The Kullback-Leibler divergence loss.
    """

    NN = nn.KLDivLoss


@instr_dcls
class BCELoss(BaseLossInstr):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    """

    NN = nn.BCELoss


@instr_dcls
class BCEWithLogitsLoss(BaseLossInstr):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    """

    NN = nn.BCEWithLogitsLoss


@instr_dcls
class SmoothL1Loss(BaseLossInstr):
    """
    Creates a criterion that uses a squared term
    if the absolute element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than torch.nn.MSELoss and in some cases
    prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    """

    NN = nn.SmoothL1Loss
