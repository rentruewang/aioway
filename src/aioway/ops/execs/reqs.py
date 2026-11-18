# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway._utils import AnyDict, any_dict, any_set
from aioway.ops.ops import Op
from aioway.t import Domain

__all__ = ["DomainReqs"]


@dcls.dataclass(frozen=True)
class DomainReqs:
    "Producer -> what its consumers require of it, where only a run time check can tell."

    reqs: AnyDict[Op, list[Domain]] = dcls.field(default_factory=any_dict)
    "The requirements of each producer `Op`."

    def __bool__(self) -> bool:
        return bool(self.reqs)

    @classmethod
    def collect(cls, root: Op) -> typing.Self:
        "Collect the requirements of every producer under `root`."

        reqs: AnyDict[Op, list[Domain]] = any_dict()
        seen_consumers = any_set()

        for consumer in root.post_order():
            # Shared `Op`s are yielded once per consumer.
            if consumer in seen_consumers:
                continue

            seen_consumers.add(consumer)

            for producer, req in zip(consumer.__deps__, consumer.__domain_deps__):
                # `>` case: too broad for the static check.
                if not producer.__domain__ > req:
                    continue

                if producer not in reqs:
                    reqs[producer] = []

                reqs[producer].append(req)

        return cls(reqs)

    def check(self, op: Op, result) -> None:
        "Check `result` of `op` against its domain, and what its consumers require of it."

        op.__domain__.check(result)

        if op not in self.reqs:
            return

        for req in self.reqs[op]:
            req.check(result)
