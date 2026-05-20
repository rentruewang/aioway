# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

import torch
from torch import nn

from aioway._types import dcls_no_repr

from .fwds import MessFwd

__all__ = [
    "ReLU",
    "ReLU6",
    "CELU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "Softmin",
    "Softmax",
    "LogSoftmax",
    "Identity",
    "Linear",
    "Bilinear",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "Embedding",
    "EmbeddingBag",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "Sequential",
]


@dcls_no_repr
class _SingleInput(MessFwd, abc.ABC):
    input: torch.Tensor


@dcls_no_repr
class ReLU(_SingleInput):
    KEY = nn.ReLU


@dcls_no_repr
class ReLU6(_SingleInput):
    KEY = nn.ReLU6


@dcls_no_repr
class CELU(_SingleInput):
    KEY = nn.CELU


@dcls_no_repr
class GELU(_SingleInput):
    KEY = nn.GELU


@dcls_no_repr
class Sigmoid(_SingleInput):
    KEY = nn.Sigmoid


@dcls_no_repr
class Tanh(_SingleInput):
    KEY = nn.Tanh


@dcls_no_repr
class Softmin(_SingleInput):
    KEY = nn.Softmin


@dcls_no_repr
class Softmax(_SingleInput):
    KEY = nn.Softmax


@dcls_no_repr
class LogSoftmax(_SingleInput):
    KEY = nn.LogSoftmax


@dcls_no_repr
class Identity(_SingleInput):
    KEY = nn.Identity


@dcls_no_repr
class Linear(_SingleInput):
    KEY = nn.Linear


@dcls_no_repr
class Bilinear(_SingleInput):
    KEY = nn.Bilinear


@dcls_no_repr
class Dropout(_SingleInput):
    KEY = nn.Dropout


@dcls_no_repr
class Dropout1d(_SingleInput):
    KEY = nn.Dropout1d


@dcls_no_repr
class Dropout2d(_SingleInput):
    KEY = nn.Dropout2d


@dcls_no_repr
class Dropout3d(_SingleInput):
    KEY = nn.Dropout3d


@dcls_no_repr
class Embedding(_SingleInput):
    KEY = nn.Embedding


@dcls_no_repr
class EmbeddingBag(_SingleInput):
    KEY = nn.EmbeddingBag


@dcls_no_repr
class BatchNorm1d(_SingleInput):
    KEY = nn.BatchNorm1d


@dcls_no_repr
class BatchNorm2d(_SingleInput):
    KEY = nn.BatchNorm2d


@dcls_no_repr
class BatchNorm3d(_SingleInput):
    KEY = nn.BatchNorm3d


@dcls_no_repr
class InstanceNorm1d(_SingleInput):
    KEY = nn.InstanceNorm1d


@dcls_no_repr
class InstanceNorm2d(_SingleInput):
    KEY = nn.InstanceNorm2d


@dcls_no_repr
class InstanceNorm3d(_SingleInput):
    KEY = nn.InstanceNorm3d


@dcls_no_repr
class Conv1d(_SingleInput):
    KEY = nn.Conv1d


@dcls_no_repr
class Conv2d(_SingleInput):
    KEY = nn.Conv2d


@dcls_no_repr
class Conv3d(_SingleInput):
    KEY = nn.Conv3d


@dcls_no_repr
class AvgPool1d(_SingleInput):
    KEY = nn.AvgPool1d


@dcls_no_repr
class AvgPool2d(_SingleInput):
    KEY = nn.AvgPool2d


@dcls_no_repr
class AvgPool3d(_SingleInput):
    KEY = nn.AvgPool3d


@dcls_no_repr
class MaxPool1d(_SingleInput):
    KEY = nn.MaxPool1d


@dcls_no_repr
class MaxPool2d(_SingleInput):
    KEY = nn.MaxPool2d


@dcls_no_repr
class MaxPool3d(_SingleInput):
    KEY = nn.MaxPool3d


@dcls_no_repr
class Sequential(_SingleInput):
    KEY = nn.Sequential
