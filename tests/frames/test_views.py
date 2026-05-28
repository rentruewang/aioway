# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._frames import TdictFrame


def test_column_attr(frame: TdictFrame) -> None:
    attrs = frame.attrs
    first_key = list(attrs.keys())[0]

    assert frame.column(first_key).attr == attrs[first_key]
    assert frame.select(first_key).attrs == {first_key: attrs[first_key]}


def test_select_attr(frame: TdictFrame) -> None:
    attrs = frame.attrs
    k_0, k_1 = list(attrs.keys())[:2]

    selected = {k_0: attrs[k_0], k_1: attrs[k_1]}
    assert frame.select(k_0, k_1).attrs == selected
