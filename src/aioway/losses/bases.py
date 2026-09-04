# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.deductions import deduction_for
from aioway.instrs import NnInstr, instr_dcls

__all__ = [
    "BaseLossInstr",
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
]

LOGGER = logging.getLogger(__name__)

_REDUCTION = frozenset(["none", "mean", "sum"])

_LOSS_TSPEC = tspecs.Unbounded(shape=torch.Size())


@instr_dcls
class BaseLossInstr(NnInstr, abc.ABC):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    @typing.override
    def module(self) -> nn.Module:
        return self.NN()


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


@deduction_for(nn.L1Loss).register
@deduction_for(nn.SmoothL1Loss).register
@deduction_for(nn.MSELoss).register
def symmetric_loss_deduct(
    self, input: tspecs.Unbounded, target: tspecs.Unbounded
) -> tspecs.Unbounded:
    if input != target:
        return NotImplemented

    return _LOSS_TSPEC
