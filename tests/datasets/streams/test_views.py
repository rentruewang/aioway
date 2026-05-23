# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.dsets import Stream


def test_column_attr(table_stream: Stream):
    attrs = table_stream.attrs
    first_key = list(attrs.keys())[0]

    assert table_stream.column(first_key).attr == attrs[first_key]
    assert table_stream.select(first_key).attrs == attrs.select(first_key)


def test_select_attr(table_stream: Stream):
    attrs = table_stream.attrs
    two_keys = list(attrs.keys())[:2]

    assert table_stream.select(*two_keys).attrs == attrs.select(*two_keys)
    assert len(attrs.select(*two_keys)) == 2
