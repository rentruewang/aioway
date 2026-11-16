# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module, on `Op` trees."

import abc
import dataclasses as dcls
import typing

from aioway._utils import any_dict
from aioway.ops import Op

__all__ = ["OpRewriter", "OpRewriteRule"]


class OpRewriteRule(typing.Protocol):
    @abc.abstractmethod
    def __call__(self, op: Op, /) -> Op:
        "The replacement of `op`, or `NotImplemented` if the rule doesn't apply."
        raise NotImplementedError


@typing.final
@dcls.dataclass(frozen=True)
class OpRewriter:
    """
    The rewriter rewrites the first `Op` (in post order) the rule applies to.

    Only the `Op`s depending on it are rebuilt, the input tree is kept.
    """

    rule: OpRewriteRule
    "The underlying rule."

    def __call__(self, root: Op, /) -> Op:
        """
        Rewrite the first `Op` the rule applies to.

        A `LookupError` would be raised if the rule applies nowhere.
        """

        for op in root.post_order():
            if (replaced := self.rule(op)) is not NotImplemented:
                return root.replace(any_dict((op, replaced)))

        raise LookupError(f"{self.rule} applies nowhere.")
