# Copyright (c) AIoWay Authors - All Rights Reserved

"The explainer based on `captum.concept`."

import abc
import dataclasses as dcls
import tempfile
import typing
from collections import abc as cabc

import torch
from captum import concept
from torch import nn
from torch.utils import data as du

from aioway.ml.tasks import Task

from ._utils import AttrAdaptor, AttrRemains

__all__ = ["CaptumConceptTask", "build_concept_dcls", "BuildTCAV"]


@typing.dataclass_transform(frozen_default=True, kw_only_default=True)
def build_concept_dcls(cls):
    return dcls.dataclass(frozen=True, kw_only=True)(cls)


# The base class ====


@dcls.dataclass
class CaptumConceptTask(Task):
    """
    The explainer based on `captum.concept.ConceptInterpreter`.

    Output: shape `[concepts, layers]`, concept's relative attention.
    """

    @build_concept_dcls
    class Strategy(abc.ABC):
        """
        Strategy for building a `captum.concept.ConceptInterpreter` for a module.

        Subclasses set `_CAPTUM_CLASS` (subclass of `ConceptInterpreter`)
        `self` must decompose to kwargs of its constructor.
        """

        _CAPTUM_CLASS: typing.ClassVar[cabc.Callable[..., concept.ConceptInterpreter]]
        "The `captum` class to build."

        def __call__(
            self, module: nn.Module, save_path: str
        ) -> concept.ConceptInterpreter:
            return self._CAPTUM_CLASS(
                model=module, save_path=save_path, **self.as_kwargs()
            )

        @abc.abstractmethod
        def score(
            self, result: dict, sets: list[list[concept.Concept]]
        ) -> torch.Tensor:
            "Each set's score on the batch, a mean over its examples. Shape (len(sets), layers)."

            raise NotImplementedError

        def as_kwargs(self) -> dict[str, typing.Any]:
            "The fields, as keyword arguments to the constructor."
            return {f.name: getattr(self, f.name) for f in dcls.fields(self)}

    cfg: Strategy
    "The concept build function."

    concepts: dict[str, du.Dataset]
    "Example inputs of each concept, by name. Bare inputs `x`, no labels."

    random: du.Dataset
    "Example inputs of no particular concept: each concept's direction is fitted against these."

    batch_size: int = 256
    "The batch size to use to iterate over the datasets."

    def __call__(self, dset: du.Dataset, module: nn.Module) -> torch.Tensor:
        """
        Each concept's score in each layer, over the whole `dset`. Shape (len(concepts), layers).

        Args:
            dset: The examples to explain, (x, y) pairs.
            module: An `nn.Module` that outputs one score per class, shape (batch, classes).
        """

        adaptor = AttrAdaptor(module)

        loaders = [
            du.DataLoader(
                cd,
                batch_size=self.batch_size,
                collate_fn=AttrRemains.collate_fn,
            )
            for cd in self.concepts.values()
        ]
        concepts = [
            concept.Concept(i, name, dl)
            for i, (name, dl) in enumerate(zip(self.concepts, loaders))
        ]
        random = concept.Concept(
            len(concepts),
            "random",
            du.DataLoader(
                self.random,
                batch_size=self.batch_size,
                collate_fn=AttrRemains.collate_fn,
            ),
        )
        sets = [[c, random] for c in concepts]

        total: torch.Tensor | float = 0.0
        n = 0

        # A new folder per current weights.
        with tempfile.TemporaryDirectory() as file:
            method = self.cfg(adaptor, file)
            loader = du.DataLoader(
                dset,
                batch_size=self.batch_size,
                collate_fn=AttrRemains.collate_fn_xy,
            )
            for *x, remains, y in loader:
                assert isinstance(remains, AttrRemains)
                result = remains.invoke(method.interpret, x, y, experimental_sets=sets)
                n += len(y)
                total += self.cfg.score(result, sets) * len(y)

        assert isinstance(total, torch.Tensor)
        return total / n


# The strategy implementations ====


def _score(layers, result: dict[str, typing.Any], k: str):
    return [float(result[k][layer]["sign_count"][0]) for layer in layers]


@build_concept_dcls
class BuildTCAV(CaptumConceptTask.Strategy):
    """
    The score: the share of examples whose own class's score rises along the concept's direction.

    TCAV passes unknown constructor kwargs on to the classifier, so every field must be a TCAV argument.
    """

    _CAPTUM_CLASS = concept.TCAV

    layers: list[str]
    "The layers in which each concept's direction is fitted."

    classifier: concept.Classifier | None = None
    "Fits the directions. `None`: Captum's default, scikit-learn's `SGDClassifier` (needs scikit-learn)."

    attribute_to_layer_input: bool = False
    "Fit the directions on each layer's input instead of its output."

    def score(self, result: dict, sets: list[list[concept.Concept]]) -> torch.Tensor:
        keys = ["-".join(str(c.id) for c in s) for s in sets]
        return torch.tensor([_score(self.layers, result, k) for k in keys])
