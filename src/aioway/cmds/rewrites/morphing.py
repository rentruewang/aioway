# Copyright (c) AIoWay Authors - All Rights Reserved

"`NetMorphLinearDeeper` on `Op` trees instead of `nn.Sequential`."

import dataclasses as dcls

from torch import nn

from aioway._utils import any_dict
from aioway.ml.rewrites.morphing import identity_init_linear
from aioway.ops import ModuleOp, Op

from .rewrites import OpRewriteRule

__all__ = ["NetMorphLinearOpDeeper", "OpTypeChain"]


@dcls.dataclass(frozen=True)
class NetMorphLinearOpDeeper(OpRewriteRule):
    """
    Deepens `Linear -> *middle -> Linear` in an `Op` tree into
    `Linear -> *middle -> identity Linear -> *middle -> Linear`.

    Like `NetMorphLinearSeqDeeper`, but on `ModuleOp`s.
    The `Op`s up to the middle are kept, so others depending on them are unchanged.
    """

    middle: tuple[type[nn.Module], ...] = (nn.ReLU,)
    "Types of the modules between the two `nn.Linear`, in order."

    def map(self, chain: list[ModuleOp]) -> Op:
        "The deeper replacement of the chain's last `Op`."

        first, *middle, last = chain
        inserted = identity_init_linear(first.nn)

        # Stack the identity and the middles again on the kept part.
        (x,) = last.__deps__
        x = _rebase(_with_module(first, inserted), x)
        for op in middle:
            x = _rebase(op, x)

        return _rebase(last, x)

    @property
    def query(self) -> OpTypeChain:
        return OpTypeChain((nn.Linear, *self.middle, nn.Linear))


@dcls.dataclass(frozen=True)
class OpTypeChain:
    """
    Search for a chain of `ModuleOp`s `a(b(c(...)))`, like `nn.Sequential`.

    Modules are matched with `isinstance`, e.g. `(nn.Linear, nn.ReLU)`.
    Each `Op` in the chain must have a single dependency.
    The chain can sit anywhere in the tree, the first one in post order is found.
    """

    pattern: tuple[type[nn.Module], ...]
    "The module types of the chain, from the innermost (applied first)."

    def __call__(self, root: Op, /) -> list[ModuleOp]:
        "The matched chain, innermost first."

        for op in root.post_order():
            if (chain := self._chain(op)) is not None:
                return chain

        raise LookupError(f"No match for {self.pattern}.")

    def _chain(self, end: Op) -> list[ModuleOp] | None:
        "The chain ending at `end`, or `None` if it doesn't match."

        reverse_chain: list[ModuleOp] = []
        op = end

        for typ in reversed(self.pattern):
            if not isinstance(op, ModuleOp) or not isinstance(op.nn, typ):
                return None

            if len(op.__deps__) != 1:
                return None

            reverse_chain.append(op)
            (op,) = op.__deps__

        return reverse_chain[::-1]


def _rebase(op: Op, dep: Op) -> Op:
    "`op` with its single dependency replaced by `dep`."
    (old,) = op.__deps__
    return op.replace(any_dict((old, dep)))


def _with_module(op: ModuleOp, module: nn.Module) -> ModuleOp:
    "`op` with its module replaced by `module`."
    (name,) = [f.name for f in dcls.fields(op) if getattr(op, f.name) is op.nn]
    return dcls.replace(op, **{name: module})
