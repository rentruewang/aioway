# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._types import track_call_count


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
