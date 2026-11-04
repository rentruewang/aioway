"""
data.py: the types. A batch is a container of tensors, in memory; a dataset is batches to read, one at a time.
Expressions take batches, statements take datasets. The state of a method is a partial container: the part known
so far, and how to get the rest.

Containers: a batch in memory, N examples side by side, data [N, *positions, *value]
  Value          one element per example: a label, a score
  Seq(causal)    elements in order: text, frames, time steps (causal: each reads only the earlier ones)
  Set            elements in no order: objects, points, the examples of a batch
  Grid(ndim)     elements on a lattice: pixels, voxels
  Graph(edges)   elements joined by edges: molecules, meshes, a model's computation
  Tree(parent)   elements under a root: parse trees, search trees
  Product(parts) several at once: Product(video, (Seq, Grid)) is frames × pixels
  what an element holds: classes · bits · real · record (from the dtype, or a torchrl spec)

Partial containers: what is known so far, and how to get the rest
  Holes[C]           C with a level per element: 0 known · 1 unknown · in between, noised to that level
  PartialSeq[S]      a list known by where it is and the step to the next: the state of any loop
  PartialTree[N, E]  a tree known by the part expanded so far (a Tree) and how it branches (options, follow)
  Dist               a set known by how to draw from it: a prediction, a policy, a generator, a search distribution
  Index              a set known by how to look things up in it: the k nearest to a query (FAISS, BM25, LSH);
                     built and filled by statements (CREATE INDEX, INSERT), searched by expressions

Results
  Path[N, E]         one walk down a PartialTree: a Seq of steps
  Example            a split's output: the model's input (Holes) and its target

Signatures
  Expr               what an expression can call on a batch, changing nothing: a method, or a model
  Model[A, B]        a learned function A → B: an nn.Module, and an Expr; a Predictor returns a Prediction
  Weights            a module whose architecture stays: an op on it changes only parameter values (a subtype)
  Codec[A, B]        encode A → B and decode back, B usually classes from a vocabulary: a tokenizer
  Step[S]            S → S, the step of a PartialSeq: a chooser of a walk, a schedule of a fill, an update
  Scan[X, S, Y]      a state carried along a sequence: (S, X) → (S, Y): an optimizer, a running average, a buffer

Datasets: read by statements, never by expressions
  Dataset[C]         batches of type C, one pass per iter(): a DataLoader, a collector, files on disk
  Buffer[C]          a dataset that statements write to as well (extend): a replay buffer
  Indexed[C]         a dataset whose examples can be fetched by position: in memory, a file of records
  Subset[C]          some examples of an Indexed, by index: what curating, splitting and sampling return; it
                     remembers what chose them, so a run can log which examples went where

scikit-learn: the API classical methods share, as protocols (anything with it fits: xgboost, lightgbm, umap)
  Estimator          fit, get_params: a model (a classifier, a regressor, a transformer, a clusterer)
  Splitter           split: index arrays for (train, test), one pair per fold
  StreamMetric       update, compute: a metric accumulated over batches (torchmetrics)
  SkModel[E]         an estimator wrapped as an Expr: its predict, transform, … take and give batches
  Environment        reset, and step a state by an action: a simulator, a game
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, NamedTuple, Protocol, Self

import torch
import torch.distributions as dist
from tensordict import TensorDictBase
from torch import Tensor, nn
from torchrl.data import Binary, Categorical, Composite, OneHot, TensorSpec

# ═══ Containers ═══════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False, repr=False)
class Container:
    """N examples, each a structure of elements; data is [N, *positions, *value]."""

    data: Any
    spec: TensorSpec | None = field(default=None, kw_only=True)

    rank: ClassVar[int] = 1  # how many dims after the first index positions

    @property
    def position_dims(self) -> int:
        return type(self).rank

    @property
    def positions(self) -> tuple[int, ...]:
        shape = self.data.batch_size if isinstance(self.data, TensorDictBase) else self.data.shape
        return tuple(shape[1 : 1 + self.position_dims])

    @property
    def value_shape(self) -> tuple[int, ...]:
        return tuple(self.spec.shape) if self.spec is not None else tuple(self.data.shape[1 + self.position_dims :])

    @property
    def type(self) -> Self:
        """The container with no examples: its structure and content, without data."""
        return self[:0]

    def with_data(self, data: Any) -> Self:
        return replace(self, data=data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: Any) -> Self:
        """Pick examples (the first dim); the structure stays."""
        return replace(self, data=self.data[i])

    def __repr__(self) -> str:
        where = " × ".join(map(str, self.positions)) + (", causal" if getattr(self, "causal", False) else "")
        return f"{type(self).__name__}({where}) of {describe(self)}"


@dataclass(frozen=True, eq=False, repr=False)
class Value(Container):
    rank: ClassVar[int] = 0


@dataclass(frozen=True, eq=False, repr=False)
class Seq(Container):
    causal: bool = False


@dataclass(frozen=True, eq=False, repr=False)
class Set(Container):
    pass


@dataclass(frozen=True, eq=False, repr=False)
class Grid(Container):
    ndim: int = 2
    rank: ClassVar[int] = 2

    @property
    def position_dims(self) -> int:
        return self.ndim


@dataclass(frozen=True, eq=False, repr=False)
class Graph(Container):
    edges: Tensor = field(default_factory=lambda: torch.zeros(2, 0, dtype=torch.long))  # [2, E], row 0 → row 1


@dataclass(frozen=True, eq=False, repr=False)
class Tree(Container):
    parent: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))  # parent[i]; −1 at a root


@dataclass(frozen=True, eq=False, repr=False)
class Product(Container):
    parts: tuple[type[Container], ...] = ()

    @property
    def position_dims(self) -> int:
        return sum(p.rank for p in self.parts)


def n_classes(spec: TensorSpec | None) -> int:
    if spec is None:
        raise TypeError("how many classes? give the container a spec, e.g. Seq(tokens, spec=Categorical(n))")
    return spec.shape[-1] if isinstance(spec, OneHot) else int(getattr(spec, "space").n)


def kind(x: Any) -> str:
    """What the elements are: "classes", "bits", "real", "record" or "distribution"; from the spec, else the dtype."""
    if isinstance(x, dist.Distribution):
        return "distribution"
    spec = x.spec if isinstance(x, Container) else x if isinstance(x, TensorSpec) else None
    if spec is None:
        data = x.data if isinstance(x, Container) else x
        if isinstance(data, TensorDictBase):
            return "record"
        assert isinstance(data, Tensor)
        return "bits" if data.dtype == torch.bool else "real" if data.is_floating_point() else "classes"
    if isinstance(spec, Composite):
        return "record"
    return "bits" if isinstance(spec, Binary) else "classes" if isinstance(spec, (Categorical, OneHot)) else "real"


def describe(x: Any) -> str:
    spec = x.spec if isinstance(x, Container) else x
    if isinstance(spec, Composite):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in spec.items()) + "}"
    k = kind(x)
    if k == "classes" and isinstance(spec, TensorSpec):
        return f"classes({n_classes(spec)})"
    shape = x.value_shape if isinstance(x, Container) else tuple(spec.shape) if spec is not None else ()
    return f"{k}{shape}" if shape else k


# ═══ Partial containers ═══════════════════════════════════════════════════════════════════════════

type Dist = dist.Distribution
"""A set known by how to draw from it: a model's prediction, a policy, a generator, a data augmentation's views,
the search distribution of evolution strategies."""

type Prediction = Dist | Tensor | TensorDictBase
"""A distribution, or what parameterizes one: logits, values, a record of them."""


@dataclass(frozen=True, eq=False)
class Holes[C: Container]:
    """C with holes: its elements, and how unknown each one is. level [N, *positions]: 0 known, 1 unknown ([MASK]
    for classes, pure noise for real values), in between noised to that level. A split makes one, fill undoes it:
      a causal Seq, every level 0     each element predicted from the ones before it (next-token prediction)
      random elements at 1            masked prediction (BERT, MAE, MaskGIT, masked diffusion)
      every element at one level t    diffusion, flow matching
      a level for each element        diffusion forcing (autoregressive and diffusion in one)
      the label at 1, the input at 0  supervised learning
    """

    known: C
    level: Tensor


@dataclass(frozen=True)
class PartialSeq[S]:
    """A list known by where it is and how it goes on: x_{i+1} = step(x_i). Every method that carries a state is a
    PartialSeq over that state; unrolling it is running the method."""

    start: S
    step: Step[S]


@dataclass(frozen=True)
class PartialTree[N, E]:
    """A tree known by the part expanded so far and how it branches. root is where it starts (prompts, states,
    positions). known is a Tree of the expanded nodes, each a record: the edge into it, the log-probability of its
    path, and whatever a method keeps there (visits and value for MCTS, keys and values for a KV cache). options(node)
    is the distribution over the edges out of a node, follow(node, edge) the node an edge leads to; an environment
    or a game adds reward and done, a critic or a verifier adds value.

    The state of every walk: greedy decoding and sampling keep one leaf per root, beam search k, best-of-N n
    independent ones, MCTS the whole subtree it has searched.
    """

    root: N
    options: Callable[[N], Dist]
    follow: Callable[[N, E], N]
    known: Tree | None = None
    reward: Callable[[N, N], Tensor] | None = None
    value: Callable[[N], Tensor] | None = None
    done: Callable[[N], Tensor] | None = None


@dataclass(frozen=True, eq=False, repr=False)
class Path(Seq):
    """One walk down a PartialTree from each root: a Seq of steps (edge, log_prob, reward, alive, done), the nodes it
    left from, and the node it reached (for an LM: the sequence)."""

    end: Any = None
    nodes: list[Any] = field(default_factory=list)


class Example(NamedTuple):
    """What a split gives: the model's input, a container with holes, and what it must predict (the hidden part,
    or a teacher's answer), with which elements count."""

    input: Holes[Any]
    target: Any
    weight: Tensor | None = None


# ═══ Signatures ═══════════════════════════════════════════════════════════════════════════════════

type Step[S] = Callable[[S], S]
"""S → S: the step of a PartialSeq. A chooser is a Step[PartialTree], a schedule a Step[Holes], an update a Step[θ]."""


class Expr:
    """What an expression can call on a batch, changing nothing and making nothing new: the catalog's expression
    methods, and a model between the statements that change it. A statement that creates a model returns an Expr,
    bound to its name: CREATE FUNCTION, after which queries call it."""


class Model[A, B](nn.Module, Expr):
    """A learned function from A to B: an nn.Module, called on containers (or on their tensors), and an Expr:
    calling it reads its parameters and changes nothing. A statement makes it (Network, Linear, …) and statements
    change it in place (Train, LoRA, …); expressions read it."""

    def forward(self, x: A, /) -> B: ...

    def __call__(self, x: A, /) -> B:
        return super().__call__(x)


type Weights = nn.Module
"""A module an operation changes only the parameter values of (quantize, init, transfer, EMA): its architecture
stays. In torch the parameters live in the module, so this is a subtype of Module rather than a separate world."""


class Codec[A, B](Protocol):
    """A way to write raw data in a vocabulary and read it back: encode A → B, decode B → A. The codec decides the
    content of B (classes of its vocabulary), and is fitted to data (BPE merges, a VQ codebook) or fixed (bytes)."""

    def encode(self, x: A, /) -> B: ...

    def decode(self, codes: B, /) -> A: ...


class Scan[X, S, Y](Protocol):
    """A state carried along a sequence of inputs: init() → S, then step(S, x) → (S, y) for each x. Optimizers (the
    moments), running averages, replay buffers, KV caches and recurrences are scans."""

    def init(self) -> S: ...

    def step(self, state: S, x: X, /) -> tuple[S, Y]: ...


class Environment(Protocol):
    """A world to act in: reset() gives starting states, step(state, action) the next state with its reward and
    whether it ended. Any simulator, a torchrl or Gym environment, a game."""

    def reset(self, n: int, /) -> Any: ...

    def step(self, state: Any, action: Tensor, /) -> Any: ...


class Index(Protocol):
    """A Set organized for lookup: the k nearest elements to each query, with their scores. How it is organized is the
    implementation: flat (exact), IVF (clusters), HNSW (a graph), PQ (compressed codes), LSH (hash buckets), an
    inverted index (BM25 over tokens), a sum tree (lookup by priority). Statements change it in place: train
    learns its structure from a sample (IVF centroids, PQ codebooks), add and remove change what it holds; search
    only reads it."""

    def train(self, sample: Value, /) -> None: ...

    def add(self, items: Set, /) -> None: ...

    def remove(self, ids: Tensor, /) -> None: ...

    def search(self, queries: Value, k: int, /) -> tuple[Set, Tensor]: ...


class Dataset[C](Protocol):
    """Batches of type C to read, a fresh pass per iter() (a DataLoader, a collector, files on disk), never a
    single-use iterator. Only statements read a dataset, as only a SQL statement's FROM scans a table; the
    expressions inside see one batch (a Container) at a time."""

    def __iter__(self) -> Iterator[C]: ...


class Buffer[C](Dataset[C], Protocol):
    """A dataset that statements write to as well: extend adds a batch, bounded by size (a replay buffer). Reading
    it samples (uniformly, by priority, the latest)."""

    def extend(self, batch: C, /) -> None: ...


class Indexed[C](Dataset[C], Protocol):
    """A dataset whose examples can be fetched by position: in memory, a file of records. Iterating gives its
    batches; take gives the batch of the examples at some indices."""

    def __len__(self) -> int: ...

    def take(self, indices: Tensor, /) -> C: ...


@dataclass(frozen=True)
class Subset[C]:
    """Some examples of an Indexed, by index, in order (an index may repeat: sampling with replacement). It is what
    curating, splitting and sampling return: they only choose indices, and a Subset is the one thing that batches,
    iterates and logs them. made_by is what chose them, from the original dataset on; a subset of a subset is
    folded into indices on the original, so its lineage is one dataset and a chain of choices."""

    parent: Indexed[C]
    indices: Tensor
    made_by: tuple[str, ...]
    batch: int = 64

    @staticmethod
    def of(data: Indexed[C], indices: Tensor, made_by: str) -> Subset[C]:
        if isinstance(data, Subset):
            return Subset(data.parent, data.indices[indices], (*data.made_by, made_by), data.batch)
        return Subset(data, indices, (made_by,))

    def __len__(self) -> int:
        return len(self.indices)

    def take(self, indices: Tensor, /) -> C:
        return self.parent.take(self.indices[indices])

    def __iter__(self) -> Iterator[C]:
        for i in range(0, len(self.indices), self.batch):
            yield self.parent.take(self.indices[i : i + self.batch])

    def __repr__(self) -> str:
        return f"{' ∘ '.join(reversed(self.made_by))} of {type(self.parent).__name__}: {len(self)} of {len(self.parent)}"


# ═══ scikit-learn: classical methods share one API ════════════════════════════════════════════════


class Estimator(Protocol):
    """A scikit-learn estimator: a config (get_params) and, once fit, a model. Classifiers and regressors predict,
    transformers transform, clusterers predict or only fit_predict."""

    def fit(self, X: Any, y: Any, /) -> Any: ...                    # y is None for an unsupervised one

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...


class Splitter(Protocol):
    """A scikit-learn cross-validation splitter (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, ShuffleSplit):
    index arrays for each (train, test)."""

    def split(self, X: Any, y: Any, groups: Any, /) -> Iterator[tuple[Any, ...]]: ...


class StreamMetric(Protocol):
    """A metric accumulated batch by batch (a torchmetrics Metric): update with each batch, compute at the end."""

    def update(self, preds: Tensor, target: Tensor, /) -> None: ...

    def compute(self) -> Tensor: ...

    def reset(self) -> None: ...


class SkModel[E: Estimator](Expr):
    """An estimator wrapped as an expression: its methods take a batch (a Value of feature vectors) and give one,
    converting to and from arrays. Calling it predicts (a transformer: transforms). A statement makes it (Sklearn)
    and fits it (FitEstimator); expressions only read it."""

    def __init__(self, estimator: E) -> None:
        self.estimator = estimator

    def __call__(self, x: Value, /) -> Value: ...

    def predict(self, x: Value, /) -> Value: ...

    def predict_proba(self, x: Value, /) -> Value: ...

    def decision_function(self, x: Value, /) -> Value: ...

    def transform(self, x: Value, /) -> Value: ...


type Predictor = Model[Any, Prediction]
"""A model that predicts: a network's output is the distribution of what it was trained to predict."""

type Schedule = Callable[[Holes[Any], Prediction], Holes[Any]]
"""A fill's step: from the holes and the model's prediction for them, the next holes (which levels fall, how far).
The catalog's Order methods are schedules."""

type Chooser = Step[PartialTree[Any, Any]]
"""A walk's step: from the tree expanded so far, the next (which edges to follow, which leaves to keep). The
catalog's Choose methods are choosers."""
