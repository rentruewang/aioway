# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._utils import AnyDict, any_dict, any_set
from aioway.ops.ops import Op
from aioway.t import Domain

__all__ = ["domain_reqs"]


def domain_reqs(root: Op) -> AnyDict[Op, list[Domain]]:
    "Producer -> what its consumers require of it, where only a run time check can tell."

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

    return reqs
