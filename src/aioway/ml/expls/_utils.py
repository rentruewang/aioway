# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn
from torch.utils import data as du

from aioway._utils import is_list_of
from aioway.t import TreeSpec, tree_flatten, tree_unflatten

__all__ = ["AttrAdaptor", "AttrRemains"]


class AttributeFunc(typing.Protocol):
    def __call__(
        self, x, additional_forward_args: tuple, target: torch.Tensor, **kwargs
    ) -> typing.Any: ...


class AttrAdaptor(nn.Module):
    """
    Adaptor for a module that takes in an `x` at runtime (`module(x)`),
    where `x` may be decomposed (via `pytree`).

    Usage:
        >>> tensors, remains = adaptor.split(x)
        >>> method.attribute(tensors, ..., additional_forward_args=(remains,))
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, *inputs) -> torch.Tensor:
        *tensors, remains = inputs
        assert isinstance(remains, AttrRemains), type(remains)
        assert is_list_of(torch.Tensor)(tensors)
        return self.module(remains.reconstruct(tensors))


@dcls.dataclass(frozen=True)
class AttrRemains:
    spec: TreeSpec
    "The spec used for reconstructing."

    tensors: cabc.Set[int]
    "The indices that are tensors."

    rest: list[typing.Any]
    "The rest of the non tensors."

    def invoke(
        self,
        func: AttributeFunc,
        x: list[torch.Tensor],
        y: torch.Tensor,
        **kwargs,
    ):
        if not is_list_of(torch.Tensor)(x):
            raise TypeError("`x` should be a list of tensors.")
        if not isinstance(y, torch.Tensor):
            raise TypeError("`y` should be a tensor.")

        return func(
            tuple(x),
            additional_forward_args=(self,),
            target=y,
            **kwargs,
        )

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
    def collate_fn_xy(cls, items: list[typing.Any]) -> tuple[typing.Any, ...]:
        """
        The collate function where `y` is involved.
        Be careful `AttrRemain` should only parse `x` or else `forward` would be messed up.

        Usage:
            >>> for *ts, r, y in du.DataLoader(dset, collate_fn=AttrRemains.collate_fn_xy):
            >>>     assert is_list_of(torch.Tensor)(ts)
            >>>     assert isinstance(r, AttrRemains)
        )
        """

        x, y = du.default_collate(items)
        tensors, remains = cls.split(x)
        return *tensors, remains, y

    @classmethod
    def collate_fn(cls, items: list[typing.Any]) -> tuple[typing.Any, ...]:
        """
        Collates the list of batchs into a tuple of `(batched_tensors, remaining)`.
        With this you don't need to call `.split` directly.

        Usage:
            >>> for *ts, r in du.DataLoader(dset, collate_fn=AttrRemains.collate_fn):
            >>>     assert is_list_of(torch.Tensor)(ts)
            >>>     assert isinstance(r, AttrRemains)
        )
        """

        batch = du.default_collate(items)
        tensors, remains = cls.split(batch)
        return *tensors, remains

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
