# Copyright (c) AIoWay Authors - All Rights Reserved

"Scores: how bad (a loss) or how good (a metric) a prediction is."

import abc
import typing

import torch
from torch import nn

from aioway.t import (
    Domain,
    FloatTensorDomain,
    IntTensorDomain,
    ScoreTensorDomain,
    TensorDomain,
)

from .ops import Op
from .thunks import ModuleOp

__all__ = [
    "ScoreOp",
    "RegressionLossOp",
    "BinaryLossOp",
    "ClassLossOp",
    "PairLossOp",
    "TripletLossOp",
    "AccuracyOp",
]


class ScoreOp(Op[torch.Tensor], abc.ABC):
    """
    How bad a prediction is (a loss), or how good (a metric): tensors in, a number out.
    """

    @typing.override
    def _infer_domain(self) -> ScoreTensorDomain:
        return ScoreTensorDomain()


class _LossOpMixin[M: nn.Module](ScoreOp, ModuleOp, abc.ABC):
    """
    A `torch.nn` loss module, applied to the input `Op`s.
    """

    module: M
    "The loss module."

    @property
    @typing.override
    def nn(self) -> M:
        return self.module


class RegressionLossOp(_LossOpMixin[nn.Module]):
    """
    `(pred: float tensor, target: float tensor) -> loss`, same shapes.

    E.g. `nn.MSELoss`, `nn.L1Loss`, `nn.HuberLoss`, `nn.SmoothL1Loss`,
    `nn.PoissonNLLLoss`, `nn.KLDivLoss`.
    """

    pred: Op[torch.Tensor]
    "The prediction."

    target: Op[torch.Tensor]
    "The target."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), FloatTensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(
            self.pred(), self.target()
        )


class BinaryLossOp(_LossOpMixin[nn.Module]):
    """
    `(pred: float tensor, target: bits or ±1) -> loss`.

    E.g. `nn.BCELoss`, `nn.BCEWithLogitsLoss`, `nn.SoftMarginLoss`,
    `nn.MultiLabelSoftMarginLoss`, `nn.HingeEmbeddingLoss`.
    """

    pred: Op[torch.Tensor]
    "The prediction (logits, probabilities or scores)."

    target: Op[torch.Tensor]
    "The target bits, or ±1."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), TensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(
            self.pred(), self.target()
        )


class ClassLossOp(_LossOpMixin[nn.Module]):
    """
    `(logits: float tensor, target: classes) -> loss`.

    E.g. `nn.CrossEntropyLoss`, `nn.NLLLoss`, `nn.MultiMarginLoss`, `nn.MultiLabelMarginLoss`.
    """

    logits: Op[torch.Tensor]
    "The logits (log-probabilities for `nn.NLLLoss`), classes on dim 1."

    target: Op[torch.Tensor]
    "The classes, or a distribution over them (soft labels, `nn.CrossEntropyLoss` only)."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), TensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(
            self.logits(), self.target()
        )


class PairLossOp(_LossOpMixin[nn.Module]):
    """
    `(a: float tensor, b: float tensor, y: ±1) -> loss`.

    E.g. `nn.MarginRankingLoss`, `nn.CosineEmbeddingLoss`.
    """

    a: Op[torch.Tensor]
    "The first of the pair."

    b: Op[torch.Tensor]
    "The second of the pair."

    y: Op[torch.Tensor]
    "1 if `a` should rank higher (or be similar), -1 otherwise."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), FloatTensorDomain(), TensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(
            self.a(),
            self.b(),
            self.y(),
        )


class TripletLossOp(_LossOpMixin[nn.Module]):
    """
    `(anchor: float tensor, positive: float tensor, negative: float tensor) -> loss`.

    E.g. `nn.TripletMarginLoss`, `nn.TripletMarginWithDistanceLoss`.
    """

    anchor: Op[torch.Tensor]
    "The anchor."

    positive: Op[torch.Tensor]
    "What should be close to `anchor`."

    negative: Op[torch.Tensor]
    "What should be far from `anchor`."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), FloatTensorDomain(), FloatTensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(
            self.anchor(),
            self.positive(),
            self.negative(),
        )


class AccuracyOp(ScoreOp):
    """
    `(logits: float tensor, target: classes) -> fraction correct`.

    Reported and selected on, never differentiated: a loss stands in for it in training.
    """

    logits: Op[torch.Tensor]
    "The logits, classes on dim 1."

    target: Op[torch.Tensor]
    "The classes."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [FloatTensorDomain(), IntTensorDomain()]

    @typing.override
    def execute(self) -> torch.Tensor:
        correct = self.logits().argmax(dim=1) == self.target()
        return correct.float().mean()
