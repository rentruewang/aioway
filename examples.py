"""
examples.py: use cases written against the interface (catalog.py), each a config like the catalog's methods: a
@dataclass(frozen=True) whose fields are its settings and whose __call__ takes the data. They show how the methods fit together; they run once the methods
are implemented.

The catalog's rule decides what kind each one is. A use case that trains or reads a whole dataset is a statement
(Stmt); one that changes what it is given is an update, a Fit like Train, and returns None; one that only reads is
an expression (Expr). The examples are not registered in the catalog: they have no title.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

import catalog as C
from catalog import Expr, Fit, Stmt
from data import Container, Dataset, Environment, Grid, Holes, Path, Predictor, Seq, Set, Value

# ═══ Statements that build a model, train it, and return it ═══════════════════════════════════════


@dataclass(frozen=True)
class Classify(Stmt):
    """Any container → a label: the types pick the layers, the label picks the loss. A statement (it trains), though
    it changes nothing it is given: the model is new."""

    steps: int = 1000
    batch: int = 64

    def __call__(self, x: Container, labels: Value, /) -> Predictor:
        model = C.Network()(x, labels)
        C.Train(steps=self.steps)(model, C.Objective(split=C.Labelled()), C.Batches(size=self.batch)(x, labels))
        return model


@dataclass(frozen=True)
class Generate(Stmt):
    """Learn any container by hiding parts of it, then generate n by filling the holes back in. The type picks the
    split (Shift, Mask or Noise); Together fills every hole at once."""

    steps: int = 1000
    fill_steps: int = 16
    n: int = 8

    def __call__(self, data: Container, /) -> Container:
        split = C.SplitFor()
        ex = split(data)
        model = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(model, C.Objective(split=split), C.Batches(size=64)(data))
        return C.Fill(steps=self.fill_steps)(C.Blank()(data[: self.n]), model, C.Together())


@dataclass(frozen=True)
class LanguageModel(Stmt):
    """Next-token prediction over a dataset, then generation as walking the LM's tree with a beam."""

    steps: int = 1000
    beam: int = 4
    length: int = 32

    def __call__(self, corpus: Dataset[Seq], prompt: Seq, /) -> Path:
        ex = C.Shift()(C.Peek()(corpus))
        lm = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(lm, C.Objective(split=C.Shift()), corpus)
        return C.Walk(steps=self.length)(C.LMTree()(lm, prompt), C.Beam(k=self.beam))


@dataclass(frozen=True)
class Plan(Stmt):
    """Learn what follows each action from experience, then act by planning inside that model."""

    steps: int = 1000
    horizon: int = 50

    def __call__(self, env: Environment, experience: Path, /) -> Path:
        batch = C.Transitions()(experience)
        ex = C.Dynamics()(batch)
        world = C.Network()(ex.input.known, ex.target)
        C.Train(steps=self.steps)(world, C.Objective(split=C.Dynamics()), C.Batches(size=64)(batch))
        tree = C.EnvTree()(env)
        return C.Walk(steps=self.horizon)(C.Planned()(tree, C.Imagined()(tree, world)), C.Greedy())


@dataclass(frozen=True)
class SearchDocuments(Stmt):
    """CREATE INDEX, then look up: training and filling the index are statements, searching it an expression."""

    kind: str = "ivf"
    k: int = 10

    def __call__(self, embed: Predictor, documents: Set, queries: Value, /) -> Set:
        vectors = Value(embed(documents))
        memory = C.EmptyIndex(kind=self.kind)(vectors)
        C.TrainIndex()(memory, vectors)
        C.Insert()(memory, C.Batches(size=1024, shuffle=False)(vectors))
        return memory.search(queries, self.k)[0]


@dataclass(frozen=True)
class Curation(Stmt):
    """Mix the sources, drop the copies, keep the better fraction, easy first: each step reads every batch."""

    weights: list[float]
    keep: float = 0.5

    def __call__(self, *sources: Dataset[Seq], quality: Callable[[Seq], Value]) -> Dataset[Seq]:
        data = C.DropNearDuplicates()(C.Dedup()(C.Mix(weights=self.weights)(*sources)))
        return C.Curriculum()(C.KeepBest(keep=self.keep)(data, quality), quality)


# ═══ Updates: they change the model they are given, and return None ═══════════════════════════════


@dataclass(frozen=True)
class ReinforcementLearning(Fit[Predictor, [Environment]]):
    """Rounds of walking the environment's tree with the policy and learning from the paths (REINFORCE with a
    batch-mean baseline). The policy changes in place."""

    rounds: int = 100
    steps: int = 10
    horizon: int = 50
    gamma: float = 0.99

    def __call__(self, policy: Predictor, env: Environment, /) -> None:
        def objective(policy: Predictor) -> Tensor:
            path = C.Walk(steps=self.horizon)(C.EnvTree()(env, policy), C.Sample())
            return C.Reinforce()(path, C.BatchBaseline()(C.DiscountedReturn(gamma=self.gamma)(path)))

        for _ in range(self.rounds):
            C.Train(steps=self.steps)(policy, objective, None)


@dataclass(frozen=True)
class Align(Fit[Predictor, [Seq, Seq]]):
    """Preferences, without a reward model: DPO against a frozen copy of the policy as it was. The policy changes in
    place."""

    steps: int = 500
    beta: float = 0.1
    batch: int = 8

    def __call__(self, policy: Predictor, chosen: Seq, rejected: Seq, /) -> None:
        reference = C.FrozenCopy()(policy)
        C.Train(steps=self.steps)(policy, C.DPOLoss(reference=reference, beta=self.beta), C.Batches(size=self.batch)(chosen, rejected))


@dataclass(frozen=True)
class Finetune(Fit[nn.Module, [Dataset[Seq]]]):
    """A pretrained model, frozen, with a low-rank update trained on new data. The model changes in place: LoRA adds
    the update, Train fits it."""

    rank: int = 8
    steps: int = 500

    def __call__(self, base: nn.Module, data: Dataset[Seq], /) -> None:
        C.LoRA(rank=self.rank)(base)
        C.Train(steps=self.steps)(base, C.Objective(split=C.Shift()), data)


# ═══ Expressions: they only read ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Inpaint(Expr):
    """Fill only some holes: the known pixels stay at level 0."""

    steps: int = 16

    def __call__(self, model: Predictor, image: Grid, missing: Tensor, /) -> Grid:
        return C.Fill(steps=self.steps)(Holes(image, missing.float()), model, C.Together())


@dataclass(frozen=True)
class DecodeWithSearch(Expr):
    """The same tree, walked four ways: one state type (PartialTree), different choosers and reshapes."""

    steps: int = 32

    def __call__(self, lm: Predictor, prompt: Seq, verify: Callable[[Tensor], Tensor], /) -> dict[str, Any]:
        tree = C.LMTree()(lm, prompt)
        walk = C.Walk(steps=self.steps)
        sampled = walk(C.TopP(p=0.9)(C.Temperature(t=0.8)(tree)), C.Sample())
        n = walk(tree, C.Independent(n=8, choose=C.Sample()))
        return {"sampled": sampled, "beam": walk(tree, C.Beam(k=4)), "best of 8": C.Best(n=8)(n, Value(verify(n.end))),
                "searched": walk(C.MCTS(simulations=64)(tree), C.Greedy())}


@dataclass(frozen=True)
class Interpret(Expr):
    """Read a layer's activations, steer along a direction, and attribute the output to the input: all reads."""

    alpha: float = 1.0

    def __call__(self, model: Predictor, layer: nn.Module, x: Container, direction: Tensor, /) -> dict[str, Any]:
        return {"activations": C.Observe()(model, layer)(x), "steered": C.Steer(alpha=self.alpha)(model, layer, direction)(x),
                "attribution": C.Attribution()(x, model)}


# ═══ Statements that fit a new model to what a model computes ════════════════════════════════════


@dataclass(frozen=True)
class Probe(Stmt):
    """Is a feature readable along one direction? Fit a linear probe to a layer's activations (closed form), and a
    sparse autoencoder to decompose them. A statement: both are fitted, though the model is only read."""

    features: int = 1024
    steps: int = 1000

    def __call__(self, model: Predictor, layer: nn.Module, x: Container, labels: Value, /) -> dict[str, Any]:
        acts = Value(C.Observe()(model, layer)(x))
        probe = C.LinearProbe()(acts, labels)
        C.LeastSquares()(probe, C.Batches(size=1024)(acts, labels))
        sae = C.SparseAutoencoder(d=acts.data.shape[-1], features=self.features)()
        C.Train(steps=self.steps)(sae, C.Objective(split=C.Reconstruction()), C.Batches(size=256)(acts))
        return {"probe": probe, "features": sae}


# ═══ Classical machine learning: the same split, a model and the statement that fits it ═════════


@dataclass(frozen=True)
class Tabular(Stmt):
    """Feature vectors and a target: gradient-boosted trees, logistic (or linear) regression and k nearest neighbours,
    each scored on held-out data. The target's type picks classification or regression for all three."""

    trees: int = 200
    k: int = 5

    def __call__(self, x: Value, y: Value, test_x: Value, test_y: Value, /) -> dict[str, Tensor]:
        data = C.Batches(size=1024)(x, y)
        boosted = C.Trees(n=self.trees)(x, y)
        C.GrowTrees()(boosted, data)
        linear = C.Linear()(x, y)
        C.Train(steps=500)(linear, C.Objective(split=C.Labelled()), data)
        memory = C.EmptyIndex()(x)
        C.Insert()(memory, C.Batches(size=1024, shuffle=False)(x))
        return {"boosted trees": C.Accuracy()(boosted(test_x), test_y), "linear": C.Accuracy()(linear(test_x), test_y),
                "kNN": C.Accuracy()(C.KNN(k=self.k)(memory, test_x, y), test_y)}


@dataclass(frozen=True)
class Cluster(Stmt):
    """No labels: project onto the principal components, then find k clusters there."""

    dim: int = 16
    k: int = 10

    def __call__(self, x: Value, /) -> Value:
        projection = C.Projection(dim=self.dim)(x)
        C.PCA()(projection, C.Batches(size=1024)(x))
        z = Value(projection(x))
        centroids = C.Centroids(k=self.k)(z)
        C.KMeans()(centroids, C.Batches(size=1024)(z))
        return centroids(z)
