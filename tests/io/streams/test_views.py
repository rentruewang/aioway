# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.io import StreamDict


def test_column_attr(table_stream: StreamDict):
    attrs = table_stream.attrs
    first_key = list(attrs.keys())[0]

    assert table_stream.column(first_key).attr == attrs[first_key]
    assert table_stream.select(first_key).attrs == {first_key: attrs[first_key]}


def test_select_attr(table_stream: StreamDict):
    attrs = table_stream.attrs
    k_0, k_1 = list(attrs.keys())[:2]

    selected = {k_0: attrs[k_0], k_1: attrs[k_1]}
    assert table_stream.select(k_0, k_1).attrs == selected
