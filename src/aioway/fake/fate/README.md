# Fate

> `Fate` stands for [f]ake [ate]n. Or [fa]ke [te]nsor. Or a tensor's [fate] (how it behaves).

Each `Fate` correspond to a `torch` core `ATen` IR.

The reason for the existence of this class is because:

1. We want to provide some additional properties, like storage costs and compute costs,
   but do not want to manage the tensor class ourselves, so by using the `__torch_dispatch__` API,
   we can reuse a lot of what `torch` did with `FakeTensorMode`.
2. Some of the operations, like boolean masking, is data dependent.
   This means that it is unsupported `FakeTensorMode`.

For case 2, we replace it with the worst case scenario.
