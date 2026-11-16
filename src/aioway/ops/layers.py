# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch
from torch import nn

from .ops import Op
from .thunks import ModuleOp

__all__ = ["LayerOp", "SameShapeOp", "ViewOp", "EmbeddingOp", "SoftmaxOp", "GridOp"]


class _UnaryModuleOpMixin[M: nn.Module](ModuleOp, abc.ABC):
    """
    A `nn.Module` applied to one input: `(input: tensor) -> tensor`.
    """

    module: M
    "The module."

    input: Op[torch.Tensor]
    "The `Op` producing the module's input."

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(self.input())

    @property
    @typing.override
    def nn(self) -> M:
        return self.module


class LayerOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer: `(input: float tensor) -> float tensor`.

    Losses are not layers, since they take an input and a target.
    """


class SameShapeOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer whose output has the input's shape:
    `(input: float tensor) -> float tensor`.

    E.g. `nn.Dropout`, `nn.LayerNorm`, `nn.BatchNorm1d`, `nn.ReLU`.
    """


class ViewOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer that views the input in another shape, values unchanged:
    `(input: tensor) -> tensor`.

    E.g. `nn.Flatten`, `nn.Unflatten`.
    """


class EmbeddingOp(_UnaryModuleOpMixin[nn.Embedding]):
    """
    A `nn.Embedding`: `(input: int tensor) -> float tensor`.
    """


class SoftmaxOp(_UnaryModuleOpMixin[nn.Softmax | nn.Softmin]):
    """
    A `nn.Softmax` or `nn.Softmin`: `(input: float tensor) -> float tensor`,
    where the output sums to 1 along `dim`.
    """

    @property
    def dim(self) -> int | None:
        "The dimension that sums to 1."
        return self.module.dim


class GridOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A convolution, pooling, padding or upsampling layer:
    `(input: float grid) -> float grid`.

    A grid is N-D: a list (1D), a list of lists (2D), and so on,
    e.g. `nn.Conv2d`, `nn.MaxPool1d`, `nn.ZeroPad2d`, `nn.Upsample`.
    """
