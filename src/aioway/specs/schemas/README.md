# Schemas

The package, schemas, is a collection of metadata describing the 'type' of data.

There are multiple supported types of metadata:

1. Shape: The shape of the tensor in a column.
2. DType: The data type of each element in the column.
3. Device: The device that the tensor lives on.

There are 2 types of schemas: `Attr` and `AttrSet` (representing schema in a table), where the latter is a collection of former.

Note:

    `AttrSet` is not called `Schema`, even though it is a table's `Schema` because:

    1. Academically, schema represents the entire database rather than just a table.
    2. `Attr` represents the "schema" of a column, and it's not called a `ColumnSchema`.
