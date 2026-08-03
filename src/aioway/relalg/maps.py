# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Exec`s that apply a transformation on the input `Exec`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import tdict_rename

from .execs import TdictExec, node_dcls

__all__ = ["MapExec", "ApplyExec", "FuncFilterExec", "RenameExec"]


@node_dcls
class MapExec(TdictExec, abc.ABC):
    """
    The shared base class for all the `map` like `Exec`s,
    which share the trait of::

        #. Having 1 child, named `source`.
        #. Calls `next` on its `source` once per `next`.
        #. Can be represented as a pure, 1 argument function.
        #. Input to output is a batch-to-batch function.

    These traits are shared in this base class.

    Note:
        Though having a 1-1 input to output batch count, this is considered to be a `flat_map`,
        where each input row can correspond to one or multiple or 0 rows, in the same minibatch.
    """

    source: TdictExec
    """
    The source stream that will be yielded from.
    """

    def __post_init__(self):
        if not isinstance(self.source, TdictExec):
            raise ValueError(
                f"{self.source=} should have been a `Stream`. Got {type(self.source)=}"
            )

    @property
    @typing.override
    def size(self) -> int:
        "This stream should have about the same length as the input."

        return self.source.size

    @abc.abstractmethod
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        """
        The protected method that subclass should overwrite.
        This method will define how each batch is processed.

        Args:
            batch: The batch to handle. Will be a `td.TensorDict`.

        Returns:
            Another `td.TensorDict`. Does not need to have the same `__len__` to the input.
            See class docstring for more details.
        """

        raise NotImplementedError

    @typing.override
    def iterate(self):
        for batch in self.source:
            batch = self._apply(batch)
            yield batch


@node_dcls
class ApplyExec(MapExec):
    """
    A `Exec` that you can customize what the `__next__` function do.

    The full loop would be something like:


    ```python
    for batch in self.source:
        yield self.apply(batch)
    ```
    """

    apply: cabc.Callable[[td.TensorDict], td.TensorDict]
    """
    Compute the output of `__next__` based on the input.
    """

    @typing.override
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        return self.apply(batch)


@node_dcls
class FuncFilterExec(MapExec):
    """
    A `Exec` that filteres on its inputs, based on a preducate function.

    The input is being used to generate predicate,
    and the output of predicate must be a boolean `torch.Tensor` of the same length as the input.

    ```python
    for batch in self.source:
        yield batch[self.predicate(batch)]
    ```
    """

    predicate: cabc.Callable[[td.TensorDict], torch.Tensor]
    """
    A function of `td.TensorDict -> torch.Tensor`.
    """

    @typing.override
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        pred = self.predicate(batch)

        if pred.dtype is not torch.bool:
            raise ValueError(
                f"Should return a boolean `torch.Tensor`. Got {pred.dtype}."
            )

        result = batch[pred]
        assert isinstance(result, td.TensorDict)
        return result


@node_dcls
class RenameExec(MapExec):
    """
    Renames some columns in the inputs in the outputs.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    Columns to rename. Mapping from original to the new names.
    """

    @typing.override
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        return tdict_rename(batch, **self.renames)
