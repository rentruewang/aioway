# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch
from torch import nn

from aioway.t import Domain, FloatTensorDomain, IntTensorDomain, TensorDomain

from .ops import Op
from .thunks import ModuleOp

__all__ = ["LayerOp", "SameShapeOp", "ViewOp", "EmbeddingOp", "SoftmaxOp", "GridOp"]


def require_specific_domain(cls):
    """
    FIXME: #769

    Marker that says: require specific domain implementation.
    """
    return cls


class _UnaryModuleOpMixin[M: nn.Module](ModuleOp, Op[torch.Tensor], abc.ABC):
    """
    A `nn.Module` applied to one input: `(input: tensor) -> tensor`.
    """

    module: M
    "The module."

    input: Op[torch.Tensor]
    "The `Op` producing the module's input."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [TensorDomain()]

    @typing.override
    def _infer_domain(self) -> TensorDomain:
        domain = self.input.__domain__

        # A super domain (e.g. `ANY`) is narrowed.
        if not isinstance(domain, TensorDomain):
            domain = TensorDomain()

        return self._layer_domain(domain)

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(self._get_dep_value(self.input))

    @property
    @typing.override
    def nn(self) -> M:
        return self.module

    @abc.abstractmethod
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        raise NotImplementedError


@require_specific_domain
class LayerOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer: `(input: float tensor) -> float tensor`.

    Losses are not layers, since they take an input and a target.
    """

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return TensorDomain()


class SameShapeOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer whose output has the input's shape:
    `(input: float tensor) -> float tensor`.

    E.g. `nn.Dropout`, `nn.LayerNorm`, `nn.BatchNorm1d`, `nn.ReLU`.
    """

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return input


@require_specific_domain
class ViewOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A `torch.nn` layer that views the input in another shape, values unchanged:
    `(input: tensor) -> tensor`.

    E.g. `nn.Flatten`, `nn.Unflatten`.
    """

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return TensorDomain()


@require_specific_domain
class EmbeddingOp(_UnaryModuleOpMixin[nn.Embedding]):
    """
    A `nn.Embedding`: `(input: int tensor) -> float tensor`.
    """

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [IntTensorDomain()]

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return FloatTensorDomain()


@require_specific_domain
class SoftmaxOp(_UnaryModuleOpMixin[nn.Softmax | nn.Softmin]):
    """
    A `nn.Softmax` or `nn.Softmin`: `(input: float tensor) -> float tensor`,
    where the output sums to 1 along `dim`.
    """

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return TensorDomain()

    @property
    def dim(self) -> int | None:
        "The dimension that sums to 1."
        return self.module.dim


@require_specific_domain
class GridOp(_UnaryModuleOpMixin[nn.Module]):
    """
    A convolution, pooling, padding or upsampling layer:
    `(input: float grid) -> float grid`.

    A grid is N-D: a list (1D), a list of lists (2D), and so on,
    e.g. `nn.Conv2d`, `nn.MaxPool1d`, `nn.ZeroPad2d`, `nn.Upsample`.
    """

    @typing.override
    def _layer_domain(self, input: TensorDomain) -> TensorDomain:
        return TensorDomain()
