# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._utils import find_common_base, track_call_count


@track_call_count
def recursive_function(i: int, history: list[int], counts: list[int]):
    if i == 0:
        return

    if i < 0:
        raise ValueError

    counts.append(recursive_function.__invoke_count__)
    history.append(i)
    recursive_function(i - 1, history, counts)


def test_invoke_count_recursive():
    assert recursive_function.__invoke_count__ == 0

    history = []
    counts = []

    recursive_function(10, history, counts)

    assert counts == list(range(1, 11))
    assert history == list(reversed(range(1, 11)))
    assert recursive_function.__invoke_count__ == 0


def _common_types_pairs():
    yield int, [1, 2, 3]
    yield object, [1, 2, 3.0]
    yield int, [1, 2, True]


@pytest.mark.parametrize("base,seq", _common_types_pairs())
def test_common_base_type(base, seq):
    assert base == find_common_base(seq)
