# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import logging
import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.dsets import InputTarget
from aioway.tspecs import TSpec

from ..deducts import TSpecInfer
from ..nn import NnInstr, NnLoss, instr_dcls

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
    def module(self) -> NnLoss:
        return NnLoss(self.NN())


@instr_dcls
class L1Loss(BaseLossInstr):
    """
    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input x and target y
    """

    NN = nn.L1Loss

    def __tspec_infer__(self):
        return SymTSpecInfer()


@instr_dcls
class MSELoss(BaseLossInstr):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y
    """

    NN = nn.MSELoss

    def __tspec_infer__(self):
        return SymTSpecInfer()


@instr_dcls
class CrossEntropyLoss(BaseLossInstr):
    """
    This criterion computes the cross entropy loss between input logits and target.
    """

    NN = nn.CrossEntropyLoss

    def __tspec_infer__(self):
        return CrossEntropyInfer()


@instr_dcls
class NLLLoss(BaseLossInstr):
    """
    The negative log likelihood loss.
    It is useful to train a classification problem with C classes.
    """

    NN = nn.NLLLoss

    def __tspec_infer__(self):
        return NllInfer()


@instr_dcls
class KLDivLoss(BaseLossInstr):
    """
    The Kullback-Leibler divergence loss.
    """

    NN = nn.KLDivLoss

    def __tspec_infer__(self):
        return KlDivInfer()


@instr_dcls
class BCELoss(BaseLossInstr):
    """
    Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
    """

    NN = nn.BCELoss

    def __tspec_infer__(self):
        return BceTSpecInfer(logits=False)


@instr_dcls
class BCEWithLogitsLoss(BaseLossInstr):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    """

    NN = nn.BCEWithLogitsLoss

    def __tspec_infer__(self):
        return BceTSpecInfer(logits=True)


@instr_dcls
class SmoothL1Loss(BaseLossInstr):
    """
    Creates a criterion that uses a squared term
    if the absolute element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than torch.nn.MSELoss and in some cases
    prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    """

    NN = nn.SmoothL1Loss

    def __tspec_infer__(self):
        return SymTSpecInfer()


class LossTSpecInfer(TSpecInfer, abc.ABC):
    def __call__(self, tspec: TSpec):
        return _LOSS_TSPEC if self._is_valid(tspec) else NotImplemented

    def _is_valid(self, tspec: TSpec) -> bool:
        if not _is_input_target(tspec):
            return False

        return self._check(tspec)

    @abc.abstractmethod
    def _check(self, tspec: tspecs.Composite) -> bool:
        raise NotImplementedError


class SymTSpecInfer(LossTSpecInfer):
    def _check(self, tspec: tspecs.Composite) -> bool:
        input = tspec["input"]
        target = tspec["target"]
        return _same_shape(tspec) and _is_unbounded(input) and _is_unbounded(target)


class KlDivInfer(LossTSpecInfer):
    def _check(self, tspec: tspecs.Composite) -> bool:
        return (
            True
            and _same_shape(tspec)
            and _is_neg_bounded(tspec["input"])
            and _is_prob(tspec["target"])
        )


class NllInfer(LossTSpecInfer):
    def _check(self, tspec: tspecs.Composite) -> bool:
        return (
            True
            and _same_shape(tspec)
            and _is_neg_bounded(tspec["input"])
            and _is_categorical(tspec["target"])
        )


class CrossEntropyInfer(LossTSpecInfer):
    def _check(self, tspec: tspecs.Composite) -> bool:
        return (
            True
            and _same_shape(tspec)
            and _is_unbounded(tspec["input"])
            and _is_categorical(tspec["target"])
        )


@dcls.dataclass
class BceTSpecInfer(LossTSpecInfer):
    logits: bool

    def _check(self, tspec: tspecs.Composite) -> bool:
        check_input = _is_unbounded if self.logits else _is_prob

        return (
            True
            and _same_shape(tspec)
            and check_input(tspec["input"])
            and _is_boolean(tspec["target"])
        )


def _same_shape(tspec: tspecs.Composite) -> bool:
    return tspec["input"].shape == tspec["target"].shape


def _is_neg_bounded(tspec: tspecs.TensorSpec) -> bool:
    # It's negative input (log of prob).
    return isinstance(tspec, tspecs.Bounded) and tspec.high == 0


def _is_categorical(tspec: tspecs.TensorSpec) -> typing.TypeIs[tspecs.Categorical]:
    return isinstance(tspec, tspecs.Categorical)


def _is_prob(tspec: tspecs.TensorSpec) -> bool:
    # Target is probability. Should also sum to 1 but now this should suffice.
    return isinstance(tspec, tspecs.Bounded) and tspec.low == 0 and tspec.high == 1


def _is_unbounded(tspec: tspecs.TensorSpec) -> typing.TypeIs[tspecs.Unbounded]:
    return isinstance(tspec, tspecs.Unbounded)


def _is_boolean(tspec: tspecs.TensorSpec) -> bool:
    return isinstance(tspec, tspecs.Categorical) and tspec.n == 2


def _is_input_target(tspec) -> typing.TypeIs[tspecs.Composite]:
    if not isinstance(tspec, tspecs.Composite):
        return False

    if tspec.data_cls is not InputTarget:
        return False

    return True
