# `Attr`s

The package, schemas, is a collection of metadata describing the 'type' of data.

There are multiple supported types of metadata:

1. `.shape`: The shape of the tensor in a column.
2. `.dtype`: The data type of each element in the column.
3. `.device`: The device that the tensor lives on.
4. `.layout`: The layout of the tensor, ususally `torch.strided` for non sparse ones.
5. `.requires_grad`: Whether or not the tensor would have a `.grad` attribute.

These 5 objects uniquely define a `torch.Tensor` (as evidenced in interal subclass maker).

There are 2 types of schemas: `Attr` and `AttrDict` (representing schema in a table), where the latter is a collection of former.
