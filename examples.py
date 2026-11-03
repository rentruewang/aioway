"""
examples.py: high-level use cases, one function each, written against the interface (catalog.py). They show how
the entries fit together; they run once the entries are implemented. Plain Python plays the script here: a
statement (train, lora, insert, curriculum, …) is a line of its own, taking datasets; expressions take batches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor, nn

import catalog as C
from data import Container, Dataset, Environment, Grid, Holes, Path, Predictor, Seq, Set, Value


def classify(x: Container, labels: Value) -> Predictor:
    """Any container → a label: the types pick the layers, the label picks the loss."""
    model = C.network(x, labels)
    C.train(model, C.objective(C.labelled), C.batches(x, labels, size=64), steps=1000)
    return model


def generate(data: Container) -> Container:
    """Learn any container by hiding parts of it, then generate one by filling the holes back in. The type picks the
    split (shift, mask or noise); the matching schedule fills (in order, most confident first, or all together)."""
    split = C.split_for(data)
    ex = split(data)
    model = C.network(ex.input.known, ex.target)
    C.train(model, C.objective(split), C.batches(data, size=64), steps=1000)
    return C.fill(C.blank(data[:8]), model, C.together, steps=16)


def inpaint(model: Any, image: Grid, missing: Tensor) -> Grid:
    """Fill only some holes: the known pixels stay at level 0."""
    return C.fill(Holes(image, missing.float()), model, C.together, steps=16)


def language_model(corpus: Dataset[Seq], prompt: Seq) -> Path:
    """Next-token prediction over a dataset, then generation as walking the LM's tree with a beam."""
    ex = C.shift(C.peek(corpus))
    lm = C.network(ex.input.known, ex.target)
    C.train(lm, C.objective(C.shift), corpus, steps=1000)
    return C.walk(C.lm_tree(lm, prompt), C.beam(4), steps=32)


def decode_with_search(lm: Any, prompt: Seq, verify: Callable[[Tensor], Tensor]) -> dict[str, Any]:
    """The same tree, walked four ways: one state type (PartialTree), different choosers and reshapes."""
    tree = C.lm_tree(lm, prompt)
    sampled = C.walk(C.top_p(C.temperature(tree, 0.8), 0.9), C.sample, steps=32)
    n = C.walk(tree, C.independent(8, C.sample), steps=32)
    return {"sampled": sampled, "beam": C.walk(tree, C.beam(4), steps=32), "best of 8": C.best(n, Value(verify(n.end)), n=8),
            "searched": C.walk(C.mcts(tree, simulations=64), C.greedy, steps=32)}


def reinforcement_learning(env: Environment, policy: Predictor, rounds: int = 100) -> Predictor:
    """Rounds of walking the environment's tree with the policy and learning from the paths (REINFORCE with a
    batch-mean baseline)."""

    def objective(policy: Predictor) -> Tensor:
        path = C.walk(C.env_tree(env, policy), C.sample, steps=50)
        return C.reinforce(path, C.batch_baseline(C.discounted_return(path, gamma=0.99)))

    for _ in range(rounds):
        C.train(policy, objective, None, steps=10)
    return policy


def plan(env: Environment, experience: Path) -> Path:
    """Learn what follows each action, then act by planning inside that model."""
    batch = C.transitions(experience)
    ex = C.dynamics(batch)
    world = C.network(ex.input.known, ex.target)
    C.train(world, C.objective(C.dynamics), C.batches(batch, size=64), steps=1000)
    tree = C.env_tree(env)
    return C.walk(C.planned(tree, C.imagined(tree, world)), C.greedy, steps=50)


def align(policy: Predictor, chosen: Seq, rejected: Seq) -> Predictor:
    """Preferences, without a reward model: DPO against a frozen copy."""
    reference = C.frozen_copy(policy)
    C.train(policy, C.dpo(reference), C.batches(chosen, rejected, size=8), steps=500)
    return policy


def finetune(base: nn.Module, data: Dataset[Seq]) -> nn.Module:
    """A pretrained model, frozen, with a low-rank update trained on new data."""
    C.lora(base, rank=8)
    C.train(base, C.objective(C.shift), data, steps=500)
    return base


def curate(sources: list[Dataset[Seq]], weights: list[float], quality: Callable[[Seq], Value]) -> Dataset[Seq]:
    """Mix the sources, drop the copies, keep the better half, easy first: each a statement reading every batch."""
    data = C.drop_near_duplicates(C.dedup(C.mix(*sources, weights=weights)))
    return C.curriculum(C.keep_best(data, quality, keep=0.5), quality)


def search_documents(embed: Predictor, documents: Set, queries: Value) -> Set:
    """CREATE INDEX, then look up: training and filling the index are statements, searching it an expression."""
    vectors = Value(embed(documents))
    memory = C.index(vectors, kind="ivf")
    C.train_index(memory, vectors)
    C.insert(memory, C.batches(vectors, size=1024, shuffle=False))
    return C.search(memory, queries, k=10)


def interpret(model: Predictor, layer: nn.Module, x: Container, labels: Value, direction: Tensor) -> dict[str, Any]:
    """Read a layer's activations, probe them, decompose them, and steer along a direction."""
    acts = Value(C.observe(model, layer)(x))
    return {"probe": C.linear_probe(acts, labels), "features": C.sparse_autoencoder(64, features=1024),
            "steered": C.steer(model, layer, direction)(x), "attribution": C.attribution(model, x)}
