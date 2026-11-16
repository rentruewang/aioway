# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module, on `Op` trees."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import any_dict
from aioway.ops import Op
from aioway.ops.thunks import ModuleOp

__all__ = ["OpRewriter", "OpRewriteRule"]


class OpRewriteRule(typing.Protocol):
    @abc.abstractmethod
    def map(self, chain: list[ModuleOp], /) -> Op:
        "Transforms a matched chain into the replacement of its last `Op`."
        ...

    @property
    def query(self) -> cabc.Callable[[Op], list[ModuleOp]]:
        "The query that matches the chain to rewrite, innermost first."
        ...


@typing.final
@dcls.dataclass(frozen=True)
class OpRewriter:
    """
    The rewriter rewrites a chain in an `Op` tree to another.

    Only the `Op`s depending on the chain are rebuilt, the input tree is kept.
    """

    rule: OpRewriteRule
    "The underlying rule."

    def __call__(self, root: Op, /) -> Op:
        """
        Match the chain, then rewrite it with the rule given.

        A `LookupError` would be raised if querying failed.
        """

        chain = self.rule.query(root)
        replaced = self.rule.map(chain)
        return root.replace(any_dict((chain[-1], replaced)))
