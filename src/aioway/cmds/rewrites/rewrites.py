# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module, on `Op` trees."

import abc
import dataclasses as dcls
import typing

from aioway.ops import Op

__all__ = ["OpRewriter", "OpRewriteRule"]


class OpRewriteRule(typing.Protocol):
    @abc.abstractmethod
    def __call__(self, root: Op, /) -> Op:
        "The rewritten tree, or `NotImplemented` if the rule applies nowhere."
        raise NotImplementedError


@typing.final
@dcls.dataclass(frozen=True)
class OpRewriter:
    """
    The rewriter rewrites an `Op` tree with the rule.

    The rule decides where (and how) to look in the tree.
    """

    rule: OpRewriteRule
    "The underlying rule."

    def __call__(self, root: Op, /) -> Op:
        """
        Rewrite the tree with the rule given.

        A `LookupError` would be raised if the rule applies nowhere.
        """

        if (result := self.rule(root)) is NotImplemented:
            raise LookupError(f"{self.rule} applies nowhere.")

        return result
