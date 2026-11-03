"""
examples.py: high-level use cases, one function each, written against the interface (catalog.py). They show how
the methods fit together; they run once the methods are implemented. A method is a config called on data:
Walk(steps=32)(tree, Beam(k=4)). Plain Python plays the script here: a statement (Train, LoRA, Insert, Curriculum,
…) is a line of its own, taking datasets; expressions take batches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor, nn

import catalog as C
from data import Container, Dataset, Environment, Grid, Holes, Path, Predictor, Seq, Set, Value


def classify(x: Container, labels: Value) -> Predictor:
    """Any container → a label: the types pick the layers, the label picks the loss."""
    model = C.Network()(x, labels)
    C.Train(steps=1000)(model, C.Objective(split=C.Labelled()), C.Batches(size=64)(x, labels))
    return model


def generate(data: Container) -> Container:
    """Learn any container by hiding parts of it, then generate one by filling the holes back in. The type picks the
    split (Shift, Mask or Noise); the matching order fills (in order, most confident first, or all together)."""
    split = C.SplitFor()
    ex = split(data)
    model = C.Network()(ex.input.known, ex.target)
    C.Train(steps=1000)(model, C.Objective(split=split), C.Batches(size=64)(data))
    return C.Fill(steps=16)(C.Blank()(data[:8]), model, C.Together())


def inpaint(model: Predictor, image: Grid, missing: Tensor) -> Grid:
    """Fill only some holes: the known pixels stay at level 0."""
    return C.Fill(steps=16)(Holes(image, missing.float()), model, C.Together())


def language_model(corpus: Dataset[Seq], prompt: Seq) -> Path:
    """Next-token prediction over a dataset, then generation as walking the LM's tree with a beam."""
    ex = C.Shift()(C.Peek()(corpus))
    lm = C.Network()(ex.input.known, ex.target)
    C.Train(steps=1000)(lm, C.Objective(split=C.Shift()), corpus)
    return C.Walk(steps=32)(C.LMTree()(lm, prompt), C.Beam(k=4))


def decode_with_search(lm: Predictor, prompt: Seq, verify: Callable[[Tensor], Tensor]) -> dict[str, Any]:
    """The same tree, walked four ways: one state type (PartialTree), different choosers and reshapes."""
    tree = C.LMTree()(lm, prompt)
    walk = C.Walk(steps=32)
    sampled = walk(C.TopP(p=0.9)(C.Temperature(t=0.8)(tree)), C.Sample())
    n = walk(tree, C.Independent(n=8, choose=C.Sample()))
    return {"sampled": sampled, "beam": walk(tree, C.Beam(k=4)), "best of 8": C.Best(n=8)(n, Value(verify(n.end))),
            "searched": walk(C.MCTS(simulations=64)(tree), C.Greedy())}


def reinforcement_learning(env: Environment, policy: Predictor, rounds: int = 100) -> Predictor:
    """Rounds of walking the environment's tree with the policy and learning from the paths (REINFORCE with a
    batch-mean baseline)."""

    def objective(policy: Predictor) -> Tensor:
        path = C.Walk(steps=50)(C.EnvTree()(env, policy), C.Sample())
        return C.Reinforce()(path, C.BatchBaseline()(C.DiscountedReturn(gamma=0.99)(path)))

    for _ in range(rounds):
        C.Train(steps=10)(policy, objective, None)
    return policy


def plan(env: Environment, experience: Path) -> Path:
    """Learn what follows each action, then act by planning inside that model."""
    batch = C.Transitions()(experience)
    ex = C.Dynamics()(batch)
    world = C.Network()(ex.input.known, ex.target)
    C.Train(steps=1000)(world, C.Objective(split=C.Dynamics()), C.Batches(size=64)(batch))
    tree = C.EnvTree()(env)
    return C.Walk(steps=50)(C.Planned()(tree, C.Imagined()(tree, world)), C.Greedy())


def align(policy: Predictor, chosen: Seq, rejected: Seq) -> Predictor:
    """Preferences, without a reward model: DPO against a frozen copy."""
    reference = C.FrozenCopy()(policy)
    C.Train(steps=500)(policy, C.DPOLoss(reference=reference), C.Batches(size=8)(chosen, rejected))
    return policy


def finetune(base: nn.Module, data: Dataset[Seq]) -> nn.Module:
    """A pretrained model, frozen, with a low-rank update trained on new data."""
    C.LoRA(rank=8)(base)
    C.Train(steps=500)(base, C.Objective(split=C.Shift()), data)
    return base


def curate(sources: list[Dataset[Seq]], weights: list[float], quality: Callable[[Seq], Value]) -> Dataset[Seq]:
    """Mix the sources, drop the copies, keep the better half, easy first: each a statement reading every batch."""
    data = C.DropNearDuplicates()(C.Dedup()(C.Mix(weights=weights)(*sources)))
    return C.Curriculum()(C.KeepBest(keep=0.5)(data, quality), quality)


def search_documents(embed: Predictor, documents: Set, queries: Value) -> Set:
    """CREATE INDEX, then look up: training and filling the index are statements, searching it an expression."""
    vectors = Value(embed(documents))
    memory = C.EmptyIndex(kind="ivf")(vectors)
    C.TrainIndex()(memory, vectors)
    C.Insert()(memory, C.Batches(size=1024, shuffle=False)(vectors))
    return memory.search(queries, 10)[0]


def interpret(model: Predictor, layer: nn.Module, x: Container, labels: Value, direction: Tensor) -> dict[str, Any]:
    """Read a layer's activations, probe them, decompose them, and steer along a direction."""
    acts = Value(C.Observe()(model, layer)(x))
    return {"probe": C.LinearProbe()(acts, labels), "features": C.SparseAutoencoder(d=64, features=1024)(),
            "steered": C.Steer()(model, layer, direction)(x), "attribution": C.Attribution()(model, x)}
