# Datasets


A dataset is a collection of `Block`s. Here are the main types of datasets:

1. `Frame`: A bounded `Chunk` like object. Supports random access.
2. `Stream`: An unbounded stream of `Chunk`s.

Where `Frame`s can make use of `Index` to make indexing faster.

Dependency wise, datasets are a layer on top of `Block`s, making use of its features.
