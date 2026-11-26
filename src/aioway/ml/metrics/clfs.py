# Copyright (c) AIoWay Authors - All Rights Reserved

"Classification metrics, from a confusion matrix."

import abc
import dataclasses as dcls
import typing

import torch
from torch import nn

from aioway.t import (
    Domain,
    IntTensorDomain,
    Pair,
    PairDomain,
    ScoreTensorDomain,
    SizedListDomain,
    TensorDomain,
)

from ._utils import as_buffer
from .metrics import Metric

__all__ = [
    "ConfusionMatrix",
    "Accuracy",
    "Precision",
    "Recall",
    "FBetaScore",
    "F1Score",
]

type _Average = typing.Literal["micro", "macro", "weighted", "none"]
"How per class scores are reduced, same as `torchmetrics`."


@dcls.dataclass(frozen=True)
class _AverageVisitor[T]:
    "What to do per `Average`. Unset ones, or ones returning `NotImplemented`, use `default`."

    default: T
    micro: T | None = None
    macro: T | None = None
    weighted: T | None = None
    none: T | None = None

    def __call__(self, avg: _Average) -> T:

        if (result := self._get(avg)) is None:
            return self.default

        else:
            return result

    def _get(self, avg: _Average) -> T | None:
        match avg:
            case "micro":
                return self.micro
            case "macro":
                return self.macro
            case "weighted":
                return self.weighted
            case "none":
                return self.none
            case _:
                raise ValueError(f"Unknown average: {avg}")


class ConfusionMatrix(Metric[Pair, torch.Tensor]):
    """
    `(preds, target) -> [C, C]` counts, rows are targets, columns are predictions.

    `preds` are logits or probabilities with classes on dim 1, or class indices.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mat = as_buffer(torch.zeros(num_classes, num_classes, dtype=torch.long))

    @typing.override
    def _input_domain(self) -> Domain[Pair]:
        return PairDomain(input=TensorDomain(), target=IntTensorDomain())

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        c = self.num_classes
        return SizedListDomain(SizedListDomain(length=c), length=c)

    @typing.override
    def _update(self, batch: Pair, /) -> None:
        preds = batch.input.detach()
        target = batch.target.detach()

        if preds.is_floating_point():
            preds = preds.argmax(dim=1)

        c = self.num_classes
        index = target.flatten() * c + preds.flatten()
        counts: typing.Any = torch.bincount(index, minlength=c * c).reshape(c, c)
        self.mat += counts

    @typing.override
    def _compute(self) -> torch.Tensor:
        return self.mat


class _StatScores(Metric[Pair, torch.Tensor], abc.ABC):
    "A score per class from the confusion matrix, then reduced by `average`."

    def __init__(self, num_classes: int, average: _Average = "macro") -> None:
        super().__init__()
        self.average = average
        self.mat = ConfusionMatrix(num_classes)

    @property
    def num_classes(self) -> int:
        "The number of classes."

        return self.mat.num_classes

    @typing.override
    def _input_domain(self) -> Domain[Pair]:
        return self.mat._input_domain()

    @typing.override
    def _output_domain(self) -> Domain[torch.Tensor]:
        visitor = _AverageVisitor[Domain[torch.Tensor]](
            default=ScoreTensorDomain(),
            none=SizedListDomain(length=self.num_classes),
        )
        return visitor(self.average)

    @typing.override
    def _update(self, batch: Pair, /) -> None:
        self.mat._update(batch)

    @typing.override
    def _compute(self) -> torch.Tensor:
        cm = self.mat.compute().float()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        scores = self._score(tp, fp, fn)
        support = tp + fn
        visitor = _AverageVisitor[torch.Tensor](
            default=scores,
            micro=self._score(tp.sum(), fp.sum(), fn.sum()),
            macro=scores.mean(),
            weighted=(scores * support).sum() / support.sum(),
        )
        return visitor(self.average)

    @abc.abstractmethod
    def _score(
        self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
    ) -> torch.Tensor:
        "The score from true positives, false positives and false negatives."

        raise NotImplementedError


class Accuracy(_StatScores):
    "The fraction correct. Per class it is recall, so `macro` is balanced accuracy."

    @typing.override
    def _score(self, tp, fp, fn):
        return safe_div(tp, tp + fn)


class Precision(_StatScores):
    "`tp / (tp + fp)`."

    @typing.override
    def _score(self, tp, fp, fn):
        return safe_div(tp, tp + fp)


class Recall(_StatScores):
    "`tp / (tp + fn)`."

    @typing.override
    def _score(self, tp, fp, fn):
        return safe_div(tp, tp + fn)


class FBetaScore(_StatScores):
    "The weighted harmonic mean of precision and recall, recall `beta` times as important."

    def __init__(
        self, num_classes: int, beta: float = 1.0, average: _Average = "macro"
    ) -> None:
        super().__init__(num_classes, average)
        self.beta = beta

    @typing.override
    def _score(self, tp, fp, fn):
        b2 = self.beta**2
        return safe_div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)


class F1Score(FBetaScore):
    "`FBetaScore` with `beta=1`."

    def __init__(self, num_classes: int, average: _Average = "macro") -> None:
        super().__init__(num_classes, beta=1.0, average=average)


# Utils ====


def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    "`num / den`, `0` where `den` is `0`, same as `torchmetrics`' `zero_division=0`."

    return torch.where(den > 0, num / den, 0.0)
