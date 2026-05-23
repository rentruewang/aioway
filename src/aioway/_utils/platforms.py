# Copyright (c) AIoWay Authors - All Rights Reserved

"Platform related utilites."

import functools
import os

__all__ = ["num_threads"]


def num_threads(target: int, ratio: float = 1) -> int:
    "If the `target` exceeds CPU count * ratio, it would be scaled down to that."

    return min(target, int(_os_cores() * ratio))


@functools.cache
def _os_cores():
    cpu = os.cpu_count()
    assert cpu
    return cpu
