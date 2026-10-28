# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import is_list_of
from aioway.t import TreeSpec, tree_flatten, tree_unflatten

__all__ = ["AttrAdaptor", "AttrRemains"]


class AttrAdaptor(nn.Module):
    """
    Adaptor for a module that takes in an `x` at runtime (`module(x)`),
    where `x` may be decomposed (via `pytree`).

    Usage:
        >>> tensors, remains = adaptor.split(x)
        >>> method.attribute(tensors, ..., additional_forward_args=remains)

    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, *inputs) -> torch.Tensor:
        *tensors, remains = inputs
        assert isinstance(remains, AttrRemains), type(remains)
        assert is_list_of(torch.Tensor)(tensors)
        return self.module(remains.reconstruct(tensors))

    def split(self, x):
        """
        A convenient adaptor to split the input,
        s.t. the helper class does not need to be imported separately.
        """

        return AttrRemains.split(x)


@dcls.dataclass(frozen=True)
class AttrRemains:
    spec: TreeSpec
    "The spec used for reconstructing."

    tensors: cabc.Set[int]
    "The indices that are tensors."

    rest: list[typing.Any]
    "The rest of the non tensors."

    def reconstruct(self, tensors: cabc.Sequence[torch.Tensor]) -> typing.Any:
        "Reconstruct from a sequence of tensors."
        if len(tensors) != len(self.tensors):
            raise ValueError("Reconstructing the wrong list of arguments.")

        leaves = []
        tensor_iter = iter(tensors)
        non_tensor_iter = iter(self.rest)

        # Reconstruct the list the same way we decompose.
        for i in range(len(tensors) + len(self.rest)):
            if i in self.tensors:
                leaves.append(next(tensor_iter))
            else:
                leaves.append(next(non_tensor_iter))

        return tree_unflatten(leaves, self.spec)

    @classmethod
    def split(cls, item) -> tuple[tuple[torch.Tensor, ...], typing.Self]:
        flatten, spec = tree_flatten(item)

        tensor_indices: set[int] = set()
        tensors: list[torch.Tensor] = []
        non_tensors: list[typing.Any] = []

        # Decompose the tensors in order.
        for i, t in enumerate(flatten):
            if isinstance(t, torch.Tensor):
                tensor_indices.add(i)
                tensors.append(t)
            else:
                non_tensors.append(t)

        inst = cls(spec=spec, tensors=tensor_indices, rest=non_tensors)
        return tuple(tensors), inst
