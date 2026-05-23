# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.dsets import Frame


def test_column_attr(frame: Frame) -> None:
    attrs = frame.attrs
    first_key = list(attrs.keys())[0]

    assert frame.column(first_key).attr == attrs[first_key]
    assert frame.select(first_key).attrs == attrs.select(first_key)


def test_select_attr(frame: Frame) -> None:
    attrs = frame.attrs
    two_keys = list(attrs.keys())[:2]

    assert frame.select(*two_keys).attrs == attrs.select(*two_keys)
    assert len(attrs.select(*two_keys)) == 2
