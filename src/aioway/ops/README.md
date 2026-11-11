# `Op`s

Like SQL, `Op`s are trees that only exist on RHS.

# Execution

Since we want to keep the design ergonomic,
every `Op` defines its own computation as a method,
and we pretend it's a simple recursion on the dependencies.

But since it's not a simple tree, some execution should cache the `Op`'s output,
so that things are not evaluated multiple times.

This means we are using a double recursion, where `execute` and `__call__` calls each other.

Now, in the simplest case, `__call__` just calls `.execute`,
but in the deduplication case, `__call__` needs to check if an `.execute` has been called,
and then use the cached result. This ensures `.execute` is only called once,
but splits `__call__` into a check.

I have thought about whether to split into plain `.execute` and `.lookup`,
but since there is no way we know statically if this is the first call without a scheduler,
as the graph can be reassembled quickly and not all the executors use schedulers,
it's best to keep a signle entry point `__call__` and keep the lookup part dynamic.
