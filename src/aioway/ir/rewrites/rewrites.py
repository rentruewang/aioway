# Copyright (c) AIoWay Authors - All Rights Reserved

"The rewriter module."

import abc
import typing

from aioway.ir import Proc
from aioway.ir.queries import Query

__all__ = ["Rewriter", "RewriteRule"]


_REWRITE_RULES: set[RewriteRule] = set()


class RewriteRule[P: Proc = typing.Any](typing.Protocol):
    @abc.abstractmethod
    def map(self, proc: P, /) -> P:
        "Transforms from a matched `Proc` to another."
        ...

    @property
    def query(self) -> Query:
        "The queries that matches the subgraph to rewrite."
        ...


@typing.final
class Rewriter[P: Proc = typing.Any]:
    """
    The rewriter rewrites a part of `Proc` to another.
    """

    rule: RewriteRule[P]

    def __call__(self, proc: P, /) -> P:
        """
        Match the subgraph, then rewrite it with the rule given.

        A `QueryError` would be raised if querying failed.
        """

        proc = proc.copy()
        matched: typing.Any = proc[self.rule.query]
        sub = self.rule.map(matched)
        proc[self.rule.query] = sub
        return proc
