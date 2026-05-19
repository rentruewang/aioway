# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

import torch
from torch import nn

from aioway._types import dcls_no_repr

from .fwds import MessFwd

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


@dcls_no_repr
class _LossFunction(MessFwd, abc.ABC):
    input: torch.Tensor


@dcls_no_repr
class L1Loss(_LossFunction):
    KEY = nn.L1Loss


@dcls_no_repr
class MSELoss(_LossFunction):
    KEY = nn.MSELoss


@dcls_no_repr
class CrossEntropyLoss(_LossFunction):
    KEY = nn.CrossEntropyLoss


@dcls_no_repr
class CTCLoss(_LossFunction):
    KEY = nn.CTCLoss


@dcls_no_repr
class NLLLoss(_LossFunction):
    KEY = nn.NLLLoss


@dcls_no_repr
class KLDivLoss(_LossFunction):
    KEY = nn.KLDivLoss


@dcls_no_repr
class BCELoss(_LossFunction):
    KEY = nn.BCELoss


@dcls_no_repr
class BCEWithLogitsLoss(_LossFunction):
    KEY = nn.BCEWithLogitsLoss


@dcls_no_repr
class SmoothL1Loss(_LossFunction):
    KEY = nn.SmoothL1Loss
