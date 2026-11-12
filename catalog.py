"""
catalog.py: every method in the original list, as a class. A method is a frozen dataclass, and says so: its fields
are its settings, its __call__ takes the data, its docstring says what it does, and the group it subclasses says
what kind of thing it is, with the group's type parameters filled in:

    @dataclass(frozen=True)
    class BestOfN(Fold[Path, Path, [Value]]):  # an expression: best-of-N is a fold, a Path in (and Values), a Path out
        n: int                              # a setting
        def __call__(self, paths: Path, scores: Value, /) -> Path: ...

    @dataclass(frozen=True)
    class Train(Fit[nn.Module, [Callable[..., Tensor], Dataset[Any] | None]]):   # a statement: it changes a module (!)
        steps: int
        def __call__(self, model: nn.Module, objective: Callable[..., Tensor], data: Dataset[Any] | None, /) -> None: ...

    BestOfN(n=16)(paths, scores)               # configure, then call: a value
    Train(steps=200)(net, loss, corpus)     # a line of its own: net changes in place, nothing is returned

Expressions take batches (a Container, in memory), build values and change nothing: queries. Statements take
datasets, or change something in place: writes, and the blocking work that needs every batch. The interface only:
bodies are left to the implementation (torch, torchrl, your own).

Each group is a base class, and its __call__ says what its methods do, in its type parameters. R is what else a
method takes (nothing by default); pyright checks every method's own __call__ against its group's.

EXPRESSIONS · on a batch: nothing changes, nothing new is made
  structure   Convert[A, B, **R]         (x: A, *R) -> B                change the container, keep the elements
              Map[A, B, **R]             (x: A, *R) -> B                change each element on its own
              Aggregate[C, **R]          (x: C, *R) -> C                each element reads the ones related to it
              Select[C, **R]             (x: C, *R) -> C                keep, drop, reorder or reweight elements
              Fold[C, A, **R]            (x: C, *R) -> A                reduce a container to one value
              Scan[X, S, Y]              init, step; (xs: Seq) -> Seq   carry a state S along a sequence
  learning    Split[C, **R]              (x: C, *R) -> Example          hide part of a container: the target
              Score[P, O, **R]           (pred: P, *R) -> O             how bad a prediction is
  partial     Start[P, **R]              (*R) -> P                      a partial container to grow
              Fill                       (holes, model, order) -> C     fill the holes, an Order deciding which
              Order                      (holes, pred) -> holes         a fill's step
              Walk                       (tree, choose) -> Path         grow a tree, a Choose deciding which leaves
              Choose                     (tree) -> tree                 a walk's step
              Unroll                     (seq) -> list                  step a state until done
              Evolve[S]                  (state: S) -> S                an unroll's step, without a gradient
              Reshape[P, **R]            (p: P, *R) -> P                how a partial container goes on
  lookup      Search[O, **R]             (index: Index, *R) -> O        what is built on index.search
  models      Wrap[M, **R]               (model: M, *R) -> Module       a view of a model: no weights of its own

STATEMENTS · making new state, changing it in place (!), or reading a whole dataset
  create      MapLayer[A, B, **R]        (*R) -> Model[A, B]            a new model: parameters of its own
              AggregateLayer[C, In, **R] (*R) -> Model[In, C]           …whose elements read the related ones
              FoldLayer[C, A, **R]       (*R) -> Model[C, A]            …that reduces a container
              Encode[A, B, **R]          (*R) -> Codec[A, B]            a new codec: raw data in a vocabulary
              Empty[S, **R]              (*R) -> S                      a new, empty index or buffer
              Copy[M]                    (x: M) -> M                    a copy with weights of its own
              Sklearn[E]                 () -> SkModel[E]               a classical model: a scikit-learn estimator
  update      Fit[X, **R]                (x: X!, *R) -> None            fit to data: a loss, a closed form, a rule
              Assign[X, **R]             (x: X!, *R) -> None            set parameter values directly
              Alter[X, **R]              (x: X!, *R) -> None            change what the parameters are
              Store[X, **R]              (x: X!, *R) -> None            train an index, insert, delete, extend
  dataset     Source[D, **R]             (*R) -> D                      where batches come from
              Curate[C, **R]             (data: Indexed[C], *R) -> Subset[C]    keep, drop, reorder, resample examples
              Partition[C, **R]          (data: Indexed[C], *R) -> splits       train / test, k folds: Subsets each
              Reduce[C, A, **R]          (data: Dataset[C], *R) -> A    one value from a whole dataset

A model is an Expr once made (data.Model is an nn.Module and an Expr): calling it reads its parameters and changes
nothing. Making one is a statement whose result is that expression, bound to its name, like CREATE FUNCTION; it is
a statement because each call makes a different one (new parameters), not because it returns a function: Given,
Guided and the views of Wrap return functions too, but make nothing new, so they stay expressions.

A statement is never a value inside an expression: in a script (program.py) it is a line of its own, as a SQL
write or CREATE INDEX sits at the top of a statement. Update statements change the argument marked ! and return
None; dataset statements change nothing but read every batch. Weights is the subtype of Module whose architecture an
op leaves alone.

So BestOfN is a Fold[Path, Path, [Value]] (a Path in and a Path out, given the Values that score it), Network a
MapLayer[Container, Prediction, [Container, Container]] (a statement: it makes a new model from examples of its
inputs and targets), Train a Fit[nn.Module, [Callable[..., Tensor], Dataset[Any] | None]] (it changes a module, reading an
objective and a dataset). Fill, Walk, Unroll and Sklearn are each one method, and the group is that method.

Supervised learning is a Supervised split and the Loss its target picks: classes make it classification, real values
regression, for any model that predicts the target (Network, Linear, KNN, a boosted-trees estimator). Classical
methods are scikit-learn's, whose API comes in a few kinds, each one of ours: an estimator is made (Sklearn) and
fitted (FitEstimator), its predict and transform are expressions, fit_predict and fit_transform read a whole dataset
(FitPredict, FitTransform), a splitter partitions one (Folds), a metric reduces one (Metric).

Curating, splitting and sampling all choose examples, so they all return a Subset: a sub-dataset that batches like
any other and remembers what chose it, so a run can log which examples each step saw.

What a statement changes is never a config: a config is a setting, and calling a create statement's config makes
the new thing (Network(d=64)(x, y), Tokenizer(vocab=32_000)()), which statements then change and expressions read.

Each method is model-agnostic (any model is an opaque function to it, or it uses none) or model-dependent (it is
a layer, or it opens a model up). Phenomena are observed rather than computed; the meta-primitives are
properties of the groups; the well-known works are programs of methods (program.py).

    python catalog.py             prints the catalog: each group's base class and __call__, each method's class,
                                  its __call__ and its settings
    python catalog.py --explain   with what each one does
    python catalog.py --short     one line per method
"""

from __future__ import annotations

import copy
import inspect
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any, ClassVar, NoDefault, ParamSpec, TypeVarTuple, get_args, get_origin

import torch
from torch import Tensor, nn

import data
from data import (
    Buffer, Chooser, Codec, Container, Dataset, Dist, Environment, Estimator, Example, Graph, Grid, Holes, Index, Indexed, Model,
    PartialSeq, PartialTree, Path, Prediction, Predictor, Schedule, Seq, Set, SkModel, Spatial, Splitter, StreamMetric, Subset,
    Temporal, Value, Weights,
)

# ═══ Methods and groups ═══════════════════════════════════════════════════════════════════════════


class Method:
    """The root of the catalog. A class with a group is a group; a class with a title is a method, registered as it
    is defined: a @dataclass(frozen=True) whose fields are its settings and whose __call__ takes the data."""

    title: ClassVar[str] = ""             # how the catalog prints a method
    state: ClassVar[str] = ""             # what it keeps, when its type carries a state
    side: ClassVar[str] = ""              # "agnostic" or "model"; the group's when empty
    changes: ClassVar[int | None] = None  # an update: which argument of __call__ it changes in place
    function: ClassVar[str] = ""          # a method torch already has: its name there (nothing to call here)
    signature: ClassVar[str] = ""         # such a method's signature, when not its group's form
    hook: ClassVar[str] = ""              # a group whose __call__ is written once: the method its methods write instead

    group: ClassVar[str] = ""             # a group: its name, its family, its form; about says what it does
    family: ClassVar[str] = ""
    form: ClassVar[str] = ""
    about: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "group" in cls.__dict__:
            _register_group(cls)
        if "title" in cls.__dict__:
            _register_method(cls)
        elif "__call__" in cls.__dict__ and any("group" in k.__dict__ for k in cls.__mro__):
            _check_use(cls)


class Expr(Method, data.Expr):
    """An expression method: on batches in memory, it changes nothing and makes nothing new. A model is an
    expression too (data.Expr), once a statement has made it."""


class Stmt(Method):
    """A statement: it makes new state (and returns it, a model being an expression bound to its name), changes its
    argument (!) in place and returns None, or reads a whole dataset. A line of its own in a script, never inside an
    expression."""


class Intent:
    """What is wanted, not which method: a config of its own class that subclasses the group it stands in for (an
    AugmentImage is a Convert[Grid, Grid]), so it type-checks wherever a method of that group goes, as a call, an
    argument or a setting. Its candidates are the configs it may become; the first is its default. A resolver lowers
    a program's intents to candidates (resolve.py); running one before then is an error."""

    candidates: ClassVar[tuple[Method, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.candidates:
            raise TypeError(f"{cls.__name__}: an intent lists its candidates")
        for c in cls.candidates:
            if issubclass(cls, Stmt) != isinstance(c, Stmt) or changes(cls) != changes(c):
                raise TypeError(f"{cls.__name__}: {type(c).__name__} is not the same kind of method (expression, statement, update)")

    def __call__(self, *args: Any, **named: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} is an intent: lower the program to concrete methods first (resolve.py)")


def settings_of(config: Method) -> dict[str, Any]:
    """A config's settings that differ from their defaults (all of the required ones)."""
    if not is_dataclass(config):
        return {}
    return {f.name: getattr(config, f.name) for f in fields(config) if _differs(getattr(config, f.name), f.default)}


def short(config: Method) -> str:
    """A config with only the settings that differ from their defaults: Beam(k=4), Greedy()."""
    return f"{type(config).__name__}({', '.join(f'{k}={v!r}' for k, v in settings_of(config).items())})"


def _differs(value: Any, default: Any) -> bool:
    if default is MISSING:
        return True
    try:
        return bool(value != default)
    except (RuntimeError, TypeError, ValueError):   # tensors and the like: compare by identity
        return value is not default


# ═══ Registry ═════════════════════════════════════════════════════════════════════════════════════

FAMILIES = {"structure": "STRUCTURE · data → data", "learning": "LEARNING · data → a training signal",
            "partial": "PARTIAL CONTAINERS · known in part, and how to get the rest",
            "lookup": "LOOKUP · a set known by how to search it", "models": "MODELS · model → a new model",
            "create": "CREATE · new state from a config: a model, a codec, an empty index or buffer, a copy",
            "update": "UPDATE · a model, a codec, an index or a buffer, changed in place",
            "dataset": "DATASET · read every batch, change nothing"}
FRAMEWORKS = {"expr": ("EXPRESSIONS · on a batch, nothing changes", ["structure", "learning", "partial", "lookup", "models"]),
              "stmt": ("STATEMENTS · making new state, changing it in place (!), or reading a whole dataset", ["create", "update", "dataset"])}

GROUPS: dict[str, list[type[Method]]] = {}   # name → its base classes (its ops, its layers), in the order defined


@dataclass(frozen=True)
class Entry:
    group: str                  # a key of GROUPS, or "phenomenon" / "meta"
    name: str                   # its title
    signature: str              # its type; for "meta", the groups it is about
    state: str                  # what it carries, and what it keeps there ("" when it carries none)
    side: str                   # "agnostic" | "model" | "" (phenomena, meta)
    function: str               # its class here, or torch's name for it
    explanation: str
    cls: type[Method] | None = None
    changes: int | None = None  # an update: which argument of __call__ it changes in place


ENTRIES: list[Entry] = []
_BY_CLASS: dict[type, Entry] = {}


def _register_group(cls: type[Method]) -> None:
    if cls.family not in FAMILIES:
        raise TypeError(f"{cls.__name__}: unknown family {cls.family!r}")
    GROUPS.setdefault(cls.group, []).append(cls)


def group_class(cls: type) -> type[Method]:
    """The group a method belongs to: the nearest class in its ancestry that declares one."""
    return next(k for k in cls.__mro__ if "group" in k.__dict__)


def _check_use(cls: type[Method]) -> None:
    """A class built on a group without being one of its methods (a use case, examples.py): it is not registered,
    but an update still returns None."""
    if group_class(cls).family == "update" and inspect.signature(cls.__call__).return_annotation not in (None, "None"):
        raise TypeError(f"{cls.__name__}: an update changes its argument in place and returns None")


def _register_method(cls: type[Method]) -> None:
    g = group_class(cls)
    stmt = issubclass(cls, Stmt)
    if stmt and g.family == "update" and cls.changes is None:
        raise TypeError(f"{cls.__name__}: an update says which argument it changes")
    if not stmt and cls.changes is not None:
        raise TypeError(f"{cls.__name__}: an expression changes nothing")
    if cls.function:                                    # torch's: nothing here to read a signature from
        signature, state = cls.signature or g.form, cls.state
    else:
        ins, out, state_type = _render(cls)
        if (cls.changes is not None) != (out == "None"):
            raise TypeError(f"{cls.__name__}: a statement that changes its argument returns None; one that does not returns a value")
        signature = " × ".join(ins) or "()" if cls.changes is not None else f"{' × '.join(ins) or '()'} → {out}"
        state = f"{state_type} — {cls.state}" if state_type and cls.state else state_type or cls.state
    doc = inspect.getdoc(cls)
    if not doc or doc == inspect.getdoc(g) and cls is not g:
        raise TypeError(f"{cls.__name__}: write what it does in its docstring")
    entry = Entry(g.group, cls.title, signature, state, cls.side or g.side, cls.function or cls.__name__, " ".join(doc.split("\n\n")[0].split()), cls, cls.changes)
    ENTRIES.append(entry)
    _BY_CLASS[cls] = entry


# ── reading a signature ──

COLLAPSE = {"Callable": "fn", "PartialTree": "PartialTree", "PartialSeq": "PartialSeq", "Step": "Step", "dict": "dict",
            "Schedule": "Schedule", "Scan": "Scan"}


def _top_level(args: str) -> list[str]:
    """Split "A, B[C, D], E" at the commas that are not inside brackets."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(args):
        depth += 1 if ch == "[" else -1 if ch == "]" else 0
        if ch == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    return out + [args[start:].strip()]


def _short(annotation: str) -> str:
    """A compact type: Container → C, nn.Module → Module, Model[A, B] → Model[A → B], Codec[A, B] → Codec[A ⇄ B],
    and Callable[...], PartialTree[...] and the like without their arguments."""
    s = re.sub(r"\b(?:data|torch|typing|nn)\.", "", annotation.replace("nn.Module", "Module"))
    out, i = "", 0
    while i < len(s):
        m = re.match(r"(\w+)\[", s[i:])
        if m and m.group(1) in (*COLLAPSE, "Model", "Codec") and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
            depth, j = 0, i + len(m.group(1))
            while True:
                depth += 1 if s[j] == "[" else -1 if s[j] == "]" else 0
                j += 1
                if depth == 0:
                    break
            name = m.group(1)
            if name in ("Model", "Codec"):
                a, b = (_short(x) for x in _top_level(s[i + len(name) + 1 : j - 1]))
                out += f"{name}[{a} {'→' if name == 'Model' else '⇄'} {b}]"
            else:
                out += COLLAPSE[name]
            i = j
        else:
            out, i = out + s[i], i + 1
    return re.sub(r"\bContainer\b", "C", out.replace("[Any]", ""))


def _type_text(t: Any) -> str:
    """A type parameter as written: a class by its name, a generic by its repr without module prefixes."""
    return t.__name__ if isinstance(t, type) and not get_args(t) else re.sub(r"\b(?:data|torch|typing|builtins|collections\.abc)\.", "", repr(t))


def _render(cls: type[Method]) -> tuple[list[str], str, str]:
    """(inputs, output, state type) of a method. A scan: Seq[X] → Seq[Y], its state S, from its Scan[X, S, Y]. Anything
    else: __call__'s positional parameters without a default (the one it changes marked !) → what it returns, a
    bounded type parameter shown as its bound."""
    if issubclass(cls, Scan):
        x, s, y = (_short(_type_text(a)) for a in _parameters(cls, Scan))
        return [f"Seq[{x}]"], f"Seq[{y}]", s
    g = group_class(cls)
    hooked = bool(g.hook) and g.hook in {n for k in cls.__mro__[: cls.__mro__.index(g)] for n in k.__dict__}
    call = getattr(cls, g.hook) if hooked else cls.__call__        # Curate's methods write indices; the group, __call__
    bounds = {t.__name__: getattr(t.__bound__, "__name__", t.__name__) for t in getattr(call, "__type_params__", ()) if t.__bound__ is not None}
    owner = g if hooked else next(k for k in cls.__mro__ if "__call__" in k.__dict__)   # an inherited call reads the class's parameters
    bounds |= {t.__name__: _type_text(a) for t, a in zip(getattr(owner, "__type_params__", ()), _parameters(cls, owner))}

    def short(annotation: Any) -> str:
        return _short(re.sub(r"\b\w+\b", lambda m: bounds.get(m.group(), m.group()), str(annotation)))

    sig = inspect.signature(call)
    params = [p for p in list(sig.parameters.values())[1:]
              if p.kind == p.VAR_POSITIONAL or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect.Parameter.empty)]
    ins = [short(p.annotation) + ("…" if p.kind == p.VAR_POSITIONAL else "") + ("!" if i == cls.changes else "") for i, p in enumerate(params)]
    returns = inspect.signature(g.__call__).return_annotation if hooked else sig.return_annotation
    return ins, short(returns), ""


def _parameters(cls: type, base: type) -> tuple[Any, ...]:
    """The type parameters cls gives base: (Path, Path) for BestOfN(Fold[Path, Path]); () when it gives none."""
    for k in cls.__mro__:
        for b in k.__dict__.get("__orig_bases__", ()):
            if get_origin(b) is base:
                return get_args(b)
    return ()


def group_parameters(cls: type) -> tuple[Any, ...]:
    """The type parameters a method gives its group: (Path, Path, (Value,)) for Best; () for a group that has none."""
    return _parameters(cls, group_class(cls))


def entry_of(f: Any) -> Entry | None:
    """The entry of a method: its class, or a config of it."""
    return _BY_CLASS.get(f if isinstance(f, type) else type(f))


def changes(f: Any) -> int | None:
    """Which argument of its __call__ f changes in place, when f is an update (None otherwise): a method, an intent
    for one, or a class of either."""
    cls = f if isinstance(f, type) else type(f)
    return cls.changes if isinstance(cls, type) and issubclass(cls, Method) else None


def is_statement(f: Any) -> bool:
    """Whether f is a statement: it makes new state, changes something or reads a dataset; never inside an expression."""
    cls = f if isinstance(f, type) else type(f)
    return isinstance(cls, type) and issubclass(cls, Stmt)


def phenomenon(name: str, explanation: str) -> None:
    ENTRIES.append(Entry("phenomenon", name, "", "", "", "", explanation))


def meta(name: str, groups: tuple[type[Method], ...], explanation: str) -> None:
    ENTRIES.append(Entry("meta", name, ", ".join(g.group for g in groups), "", "", "", explanation))


# ═══ Expression groups ════════════════════════════════════════════════════════════════════════════
# Each group's __call__ is what its methods do, in the group's own type parameters: Fold[C, A] takes a C and gives
# an A. R is what else a method takes (none by default), so Best, a Fold[Path, Path, [Value]], takes a Path and a
# Value. A method's own __call__ names its inputs; pyright checks it against its group's. An expression makes no new
# state: building a model (or a codec, an index, a buffer) is a statement, CREATE.


class Convert[A, B, **R = []](Expr):
    """Change the container, keep what the elements are."""
    group, family, form, side = "convert", "structure", "C[a] → C′[a]", "agnostic"

    def __call__(self, x: A, /, *rest: R.args, **named: R.kwargs) -> B: ...


class Map[A, B, **R = []](Expr):
    """Change each element on its own, keep the container."""
    group, family, form, side = "map", "structure", "C[a] → C[b]", "agnostic"

    def __call__(self, x: A, /, *rest: R.args, **named: R.kwargs) -> B: ...


class Aggregate[C, **R = []](Expr):
    """Each element reads the elements it is related to: a C in, a C out."""
    group, family, form, side = "aggregate", "structure", "C[h] → C[h]", "agnostic"

    def __call__(self, x: C, /, *rest: R.args, **named: R.kwargs) -> C: ...


class Select[C, **R = []](Expr):
    """Keep, drop, reorder or reweight elements: a C in, a C out."""
    group, family, form, side = "select", "structure", "C[a] → C[a]", "agnostic"

    def __call__(self, x: C, /, *rest: R.args, **named: R.kwargs) -> C: ...


class Fold[C, A, **R = []](Expr):
    """Reduce a container C to one value A."""
    group, family, form, side = "fold", "structure", "C[a] → a", "agnostic"

    def __call__(self, x: C, /, *rest: R.args, **named: R.kwargs) -> A: ...


class Scan[X, S, Y](Expr):
    """Carry a state along a sequence: init gives the first state, step takes a state and an element to the next
    state and an output; calling it runs the steps along a sequence (from the end when reverse)."""
    group, family, form, side = "scan", "structure", "Seq[X] → Seq[Y]", "agnostic"
    reverse: ClassVar[bool] = False

    def init(self) -> S: ...

    def step(self, state: S, x: X, /) -> tuple[S, Y]: ...

    def __call__(self, xs: Seq, /) -> Seq: ...


class Split[C, **R = []](Expr):
    """Hide part of a container C; what was hidden (or a teacher's answer) is the target."""
    group, family, form, side = "split", "learning", "C → Example", "agnostic"

    def __call__(self, x: C, /, *rest: R.args, **named: R.kwargs) -> Example: ...


class Score[P, O = Tensor, **R = []](Expr):
    """How bad a prediction P is (against R: a target, a weight): every loss, every surrogate objective, every metric.
    O is a number to minimize (a Tensor), or one per example (a Value)."""
    group, family, form, side = "score", "learning", "Prediction × C → ℝ", "agnostic"

    def __call__(self, pred: P, /, *rest: R.args, **named: R.kwargs) -> O: ...


class Start[P, **R = []](Expr):
    """A partial container P to grow, from R: a tree for a walk, holes for a fill. (An empty index or buffer is new
    state: Empty, a statement.)"""
    group, family, form, side = "start", "partial", "… → Partial", "agnostic"

    def __call__(self, *args: R.args, **named: R.kwargs) -> P: ...


@dataclass(frozen=True)
class Fill(Expr):
    """Repeatedly predict the holes and lower their levels as the order decides (generation, inpainting,
    denoising), until every level is 0. With levels in (0, 1), what the model predicts is the flow back to the data.
    steps is how many predictions."""
    group, family, form, side = "fill", "partial", "Holes[C] → C", "agnostic"
    about = "fill the holes, an order deciding which and how fast: a split, undone"
    title = "Fill"
    state = "Holes[C] — the levels fall, as the order decides"
    steps: int

    def __call__[C: Container](self, holes: Holes[C], model: Predictor, order: Schedule, /) -> C: ...


class Order(Expr):
    """Which holes a fill fills next, and how far: a fill's schedule, from the holes and the model's prediction."""
    group, family, form, side = "order", "partial", "Holes × Prediction → Holes", "agnostic"

    def __call__(self, holes: Holes[Any], pred: Prediction, /) -> Holes[Any]: ...


@dataclass(frozen=True)
class Walk(Expr):
    """Grow the known part of a tree from its roots, steps times, a chooser deciding which edges to follow and
    which leaves to keep; then the path to a kept leaf. Sampling, greedy decoding, beam search, best-of-N and RL
    rollouts are walks with different choosers."""
    group, family, form, side = "walk", "partial", "PartialTree → Path", "agnostic"
    about = "grow the known part of a tree, a chooser deciding which leaves to keep, then read a path"
    title = "Walk"
    state = "PartialTree — its known part, grown by the chooser"
    steps: int

    def __call__(self, tree: PartialTree[Any, Any], choose: Chooser, /) -> Path: ...


class Choose(Expr):
    """A walk's step: from the tree expanded so far, the next (which edges to follow, which leaves to keep)."""
    group, family, form, side = "choose", "partial", "PartialTree → PartialTree", "agnostic"

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class Unroll(Expr):
    """The list a PartialSeq stands for: start, step(start), step(step(start)), … steps of them."""
    group, family, form, side = "unroll", "partial", "PartialSeq[S] → S", "agnostic"
    about = "step a state until done"
    title = "Unroll"
    state = "S — any state"
    steps: int

    def __call__(self, seq: PartialSeq[Any], /) -> list[Any]: ...


class Evolve[S](Expr):
    """An unroll's step, S → S, where there is no gradient to follow or the gradient is followed with noise."""
    group, family, form, side = "evolve", "partial", "S → S", "agnostic"

    def __call__(self, state: S, /) -> S: ...


class Reshape[P, **R = []](Expr):
    """Change how a partial container (or a model) P goes on: its options, its model. A P in, a P out."""
    group, family, form, side = "reshape", "partial", "Partial → Partial", "agnostic"

    def __call__(self, p: P, /, *rest: R.args, **named: R.kwargs) -> P: ...


class Search[O, **R = []](Expr):
    """What is built on looking up an index (index.search): the k elements nearest each query, put to use as an O."""
    group, family, form, side = "search", "lookup", "Index × Value → Set", "agnostic"

    def __call__(self, index: Index, /, *rest: R.args, **named: R.kwargs) -> O: ...


class Wrap[M, **R = []](Expr):
    """A view of a model M: a module around it that reads its parameters (and adds none), M staying as it was. A copy
    with weights of its own is new state: Copy, a statement."""
    group, family, form, side = "wrap", "models", "Module → Module′", "agnostic"

    def __call__(self, model: M, /, *rest: R.args, **named: R.kwargs) -> nn.Module: ...


# ═══ Statement groups ═════════════════════════════════════════════════════════════════════════════
# A statement that makes new state from a config and returns it: a model with parameters of its own, a codec, an
# empty index or buffer, a copy. CREATE TABLE, in a database: each call makes a different one, so it is a line of
# its own and the name it is bound to is the one thing that refers to it.


class MapLayer[A, B, **R = []](Stmt):
    """Create a map: a model A → B with parameters of its own, its sizes read from examples like the data it will see."""
    group, family, form, side = "layer", "create", "builds C[a] → C[b]", "model"

    def __call__(self, *like: R.args, **named: R.kwargs) -> Model[A, B]: ...


class AggregateLayer[C, In = C, **R = []](Stmt):
    """Create an aggregate: a model whose elements read the ones they are related to, a C out (reading a C, unless In
    says what else it reads: a memory, a second container)."""
    group, family, form, side = "layer", "create", "builds C[h] → C[h]", "model"

    def __call__(self, *like: R.args, **named: R.kwargs) -> Model[In, C]: ...


class FoldLayer[C, A, **R = []](Stmt):
    """Create a fold: a model that reduces a C to an A."""
    group, family, form, side = "layer", "create", "builds C[a] → a", "model"

    def __call__(self, *like: R.args, **named: R.kwargs) -> Model[C, A]: ...


class Encode[A, B, **R = []](Stmt):
    """Create a codec: raw data A written in a vocabulary B, and read back. One that learns is fitted by FitCodec."""
    group, family, form, side = "encode", "create", "builds A ⇄ C[classes]", "model"

    def __call__(self, *like: R.args, **named: R.kwargs) -> Codec[A, B]: ...


class Empty[S, **R = []](Stmt):
    """Create an empty store S, to fill with statements: an index (its sizes from R), a buffer."""
    group, family, form, side = "empty", "create", "… → Index | Buffer", "agnostic"

    def __call__(self, *args: R.args, **named: R.kwargs) -> S: ...


class Copy[M](Stmt):
    """Create a copy of M with weights of its own, as M is now: a snapshot that later changes to M do not reach."""
    group, family, form, side = "copy", "create", "Module → Module", "agnostic"

    def __call__(self, x: M, /) -> M: ...


@dataclass(frozen=True)
class Sklearn[E: Estimator](Stmt):
    """Create a classical model from a scikit-learn estimator, or anything with its API (xgboost, lightgbm, umap): a
    fresh copy each call, so each is a model of its own. FitEstimator fits it; then its predict, predict_proba,
    transform, … are expressions on batches."""
    group, family, form, side = "estimator", "create", "Estimator → SkModel", "agnostic"
    about = "a scikit-learn estimator, as a model: fitted by a statement, then called on batches"
    title = "scikit-learn estimator"
    estimator: E

    def __call__(self) -> SkModel[E]:
        return SkModel(copy.deepcopy(self.estimator))


# An update changes its first argument, an X, in place and returns None; R is what else it reads.


class Fit[X, **R = []](Stmt):
    """Fit an X to data, in place: a model by a loss's gradient or a closed form, a codec or an estimator by its own
    rule."""
    group, family, form, side = "fit", "update", "X! × Dataset", "agnostic"
    changes = 0

    def __call__(self, x: X, /, *rest: R.args, **named: R.kwargs) -> None: ...


class Assign[X, **R = []](Stmt):
    """Set an X's parameter values directly, without a gradient: copied in, initialized, averaged."""
    group, family, form, side = "assign", "update", "Weights!", "agnostic"
    changes = 0

    def __call__(self, x: X, /, *rest: R.args, **named: R.kwargs) -> None: ...


class Alter[X, **R = []](Stmt):
    """Change what an X's parameters are: add some, freeze some, quantize, shard."""
    group, family, form, side = "alter", "update", "Module!", "model"
    changes = 0

    def __call__(self, x: X, /, *rest: R.args, **named: R.kwargs) -> None: ...


class Store[X, **R = []](Stmt):
    """Keep a store X up to date: learn an index's structure, insert into it, delete from it, add to a buffer."""
    group, family, form, side = "store", "update", "Index! / Buffer!", "agnostic"
    changes = 0

    def __call__(self, x: X, /, *rest: R.args, **named: R.kwargs) -> None: ...


class Source[D, **R = []](Stmt):
    """Where batches come from, a dataset D: memory, an environment, a buffer, an expression on each batch."""
    group, family, form, side = "source", "dataset", "… → Dataset", "agnostic"

    def __call__(self, *args: R.args, **named: R.kwargs) -> D: ...


class Curate[C, **R = []](Stmt):
    """Keep, drop, reorder or resample examples, looking at all of them (blocking): a sub-dataset of the one given.
    A method only chooses the indices; the group makes them a Subset, which batches, iterates and logs them."""
    group, family, form, side = "curate", "dataset", "Indexed → Subset", "agnostic"
    hook = "indices"

    def indices(self, data: Indexed[C], /, *rest: R.args, **named: R.kwargs) -> Tensor: ...

    def __call__(self, data: Indexed[C], /, *rest: R.args, **named: R.kwargs) -> Subset[C]:
        return Subset.of(data, self.indices(data, *rest, **named), short(self))


class Partition[C, **R = []](Stmt):
    """Split a dataset into sub-datasets: a list of splits, each a tuple of parts (train and test, or train,
    validation and test). A method only chooses the indices of each part; the group makes them Subsets."""
    group, family, form, side = "partition", "dataset", "Indexed → splits of Subsets", "agnostic"
    hook = "parts"

    def parts(self, data: Indexed[C], /, *rest: R.args, **named: R.kwargs) -> tuple[tuple[Tensor, ...], ...]: ...

    def __call__(self, data: Indexed[C], /, *rest: R.args, **named: R.kwargs) -> tuple[tuple[Subset[C], ...], ...]:
        return tuple(tuple(Subset.of(data, part, f"{short(self)}[{i}][{j}]") for j, part in enumerate(split))
                     for i, split in enumerate(self.parts(data, *rest, **named)))


class Reduce[C, A, **R = []](Stmt):
    """One value A from a whole dataset of C: a metric over every batch, a statistic, one batch."""
    group, family, form, side = "reduce", "dataset", "Dataset → a", "agnostic"

    def __call__(self, data: Dataset[C], /, *rest: R.args, **named: R.kwargs) -> A: ...


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# EXPRESSIONS · on a batch, nothing changes, nothing new is made
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ═══ Convert[A, B, **R = []]: C[a] → C′[a] ════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Patchify(Convert[Grid, Grid]):
    """An image → a coarser grid of p×p patches, each one element (a reshape); a linear map after it is a patch
    embedding."""
    title = "Patching"
    size: int

    def __call__(self, x: Grid, /) -> Grid: ...


@dataclass(frozen=True)
class Flatten(Convert[Grid, Seq]):
    """A grid → a sequence in raster order: the lattice is forgotten, learned positions say where each element was."""
    title = "Flattening"

    def __call__(self, x: Grid, /) -> Seq: ...


@dataclass(frozen=True)
class Concat(Convert[Seq, Seq]):
    """Several sequences of the same width as one (image patches, then caption tokens): one network reads them all."""
    title = "Multimodal sequence"

    def __call__(self, *parts: Seq) -> Seq: ...


@dataclass(frozen=True)
class RandomFlip(Convert[Grid, Grid]):
    """Mirror each image with probability p: its elements move, none changes."""
    title = "Data augmentation: random flip"
    p: float = 0.5

    def __call__(self, x: Grid, /) -> Grid: ...


@dataclass(frozen=True)
class RandomCrop(Convert[Grid, Grid]):
    """Pad, and cut back at a random offset: a random translation."""
    title = "Data augmentation: random crop"
    padding: int = 2

    def __call__(self, x: Grid, /) -> Grid: ...


@dataclass(frozen=True)
class Transitions(Convert[Path, Value]):
    """The steps of walked paths as a set of transitions (state, action, reward, next state, done, and the action's
    log-probability), their order dropped: what a replay buffer holds and RL losses read."""
    title = "Experience (transitions)"

    def __call__(self, path: Path, returns: Seq | None = None, /) -> Value: ...


@dataclass(frozen=True)
class PathTo(Convert[PartialTree[Any, Any], Path, [Tensor]]):
    """The steps from the root to a leaf of a tree's known part: the result of a walk."""
    title = "A path from a tree"

    def __call__(self, tree: PartialTree[Any, Any], leaf: Tensor, /) -> Path: ...


@dataclass(frozen=True)
class FewShot(Convert[Seq, Seq]):
    """Solved examples, then the query, as one prompt: the model adapts by reading them."""
    title = "In-context learning (few-shot prompts)"
    function, signature = "Concat", "Seq… → Seq"


# ═══ Map[A, B, **R = []]: C[a] → C[b] ═════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RoPE(Map[Tensor, Tensor, [Tensor]]):
    """Rotate pairs of query and key features by angles proportional to the position, so that attention depends only
    on relative offsets."""
    title = "Positional encoding: rotary (RoPE)"

    def __call__(self, x: Tensor, positions: Tensor, /) -> Tensor: ...


@dataclass(frozen=True)
class VectorQuantize(Map[Container, Container, [Tensor]]):
    """Each element → the id of its nearest codebook entry (real → classes), with a straight-through gradient."""
    title = "Vector quantization"

    def __call__(self, x: Container, codebook: Tensor, /) -> Container: ...


@dataclass(frozen=True)
class StraightThrough(Map[Tensor, Tensor, [Tensor]]):
    """Forward the hard value, backpropagate through the soft one: a gradient for a choice that has none."""
    title = "Straight-through estimator"
    side = "agnostic"

    def __call__(self, hard: Tensor, soft: Tensor, /) -> Tensor: ...


@dataclass(frozen=True)
class Condition(Map[Container, Container, [Value]]):
    """Attach a label to every element (its one-hot vector next to each element's values; a none label for guidance)."""
    title = "Conditioning"
    side = "agnostic"

    def __call__(self, x: Container, label: Value, /) -> Container: ...


@dataclass(frozen=True)
class ColorJitter(Map[Grid, Grid]):
    """Scale each image's brightness and contrast by random factors."""
    title = "Data augmentation: color jitter"
    side = "agnostic"
    strength: float = 0.2

    def __call__(self, x: Grid, /) -> Grid: ...


@dataclass(frozen=True)
class GaussianNoise(Map[Container, Container]):
    """x + σ·ε at every element."""
    title = "Data augmentation: additive noise"
    side = "agnostic"
    sigma: float = 0.05

    def __call__(self, x: Container, /) -> Container: ...


@dataclass(frozen=True)
class LabelSmoothing(Map[Value, Value]):
    """A class → a distribution over classes, (1 − ε)·one_hot + ε / n: never pushed to infinite confidence."""
    title = "Label smoothing"
    side = "agnostic"
    eps: float = 0.1

    def __call__(self, y: Value, /) -> Value: ...


@dataclass(frozen=True)
class Attribution(Map[Container, Container, [Predictor]]):
    """How much each input element pushed the output: the input times the output's gradient, on any differentiable
    model."""
    title = "Attribution (input × gradient)"
    side = "agnostic"

    def __call__(self, x: Container, model: Predictor, /) -> Container: ...


@dataclass(frozen=True)
class Chinchilla(Map[float, tuple[float, float]]):
    """A compute budget → (parameters, tokens): C ≈ 6·N·D with D ≈ 20 tokens per parameter."""
    title = "Compute-optimal allocation (Chinchilla)"
    side = "agnostic"

    def __call__(self, compute: float, /) -> tuple[float, float]: ...


# ═══ Aggregate[C, **R = []]: C[h] → C[h] ══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Mixup(Aggregate[tuple[Container, Value]]):
    """Each example blended with a random partner of the batch, labels by the same λ: the batch as a Set, paired at
    random."""
    title = "Data augmentation: mixup"
    side = "agnostic"
    alpha: float = 0.2

    def __call__(self, pair: tuple[Container, Value], /) -> tuple[Container, Value]: ...


@dataclass(frozen=True)
class Cutmix(Aggregate[tuple[Grid, Value]]):
    """A random box pasted from a partner image; the labels mix by its area."""
    title = "Data augmentation: cutmix"
    side = "agnostic"

    def __call__(self, pair: tuple[Grid, Value], /) -> tuple[Grid, Value]: ...


@dataclass(frozen=True)
class MeanBaseline(Aggregate[Value]):
    """Each return minus the batch's mean: an advantage with less variance and no bias."""
    title = "Baseline: the batch mean"
    side = "agnostic"

    def __call__(self, returns: Value, /) -> Value: ...


@dataclass(frozen=True)
class GroupAdvantage(Aggregate[Value]):
    """Each reward standardized within its group of n samples from one prompt: an advantage without a critic."""
    title = "GRPO (group-relative advantages)"
    side = "agnostic"
    n: int

    def __call__(self, rewards: Value, /) -> Value: ...


# ═══ Select[C, **R = []]: C[a] → C[a] ═════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class QualityFilter(Select[Container, [Value]]):
    """Keep the examples of a batch scoring above a threshold (a classifier's, a verifier's, a perplexity). Keeping
    the best fraction needs every batch: KeepBest."""
    title = "Data filtering"
    threshold: float

    def __call__[C: Container](self, x: C, scores: Value, /) -> C: ...


@dataclass(frozen=True)
class TokenDropout(Select[Seq]):
    """Drop a random fraction of the positions; the rest keep their order."""
    title = "Data augmentation: token dropout"
    p: float = 0.1

    def __call__(self, x: Seq, /) -> Seq: ...


# ═══ Fold[C, A, **R = []]: C[a] → a ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DecayGroups(Fold[nn.Module, list[dict[str, Any]]]):
    """The parameters to decay (the weight matrices) apart from the rest (norms, biases, embeddings)."""
    title = "Weight decay (parameter groups)"
    weight_decay: float = 0.1

    def __call__(self, model: nn.Module, /) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class MuP(Fold[nn.Module, list[dict[str, Any]]]):
    """Hidden matrices reweighted by base_width / width, so a learning rate tuned narrow carries over wide."""
    title = "Parameterization (μP)"
    width: int
    base_width: int

    def __call__(self, model: nn.Module, /) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DiscountedReturn(Fold[Path, Value]):
    """Σ γ^t·r_t along each path: what RL maximizes. A walk can't be differentiated through, so its gradient comes
    from scoring the path (reinforce)."""
    title = "Expected return (the RL objective)"
    gamma: float = 1.0

    def __call__(self, path: Path, /) -> Value: ...


@dataclass(frozen=True)
class BestOfN(Fold[Path, Path, [Value]]):
    """The best-scoring of each group of n paths (scored by a verifier or a reward model)."""
    title = "Best-of-N with a verifier"
    n: int

    def __call__(self, paths: Path, scores: Value, /) -> Path: ...


@dataclass(frozen=True)
class MajorityVote(Fold[Value, Value]):
    """The most frequent answer of each group of n samples."""
    title = "Self-consistency (majority vote)"
    n: int

    def __call__(self, answers: Value, /) -> Value: ...


@dataclass(frozen=True)
class EarlyStop(Fold[Sequence[Tensor], bool]):
    """Whether the validation scores, lowest best, have not improved on their best for `patience` evaluations."""
    title = "Early stopping"
    patience: int = 3

    def __call__(self, curve: Sequence[Tensor], /) -> bool: ...


@dataclass(frozen=True)
class ScalingLaw(Fold[Seq, tuple[float, float], [Seq]]):
    """loss ≈ a·size^(−b), fitted on the logs."""
    title = "Scaling laws (power-law fit)"

    def __call__(self, sizes: Seq, losses: Seq, /) -> tuple[float, float]: ...


# ═══ Scan[X, S, Y]: Seq[X] → Seq[Y] ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Optimizer(Scan[Tensor, Tensor, Tensor]):
    """Gradients in, updates out: SGD carries nothing, momentum a velocity, Adam two moments, Muon orthogonalizes.
    A setting of Train, which carries its state from step to step."""
    title = "Optimizers (SGD, momentum, Adam, Muon)"
    state = "the moments of every parameter"
    kind: str = "adamw"
    lr: float = 3e-4

    def init(self) -> Tensor: ...

    def step(self, state: Tensor, grad: Tensor, /) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class WarmupCosine(Scan[int, int, float]):
    """A learning rate for each step: linear warmup, then cosine decay. A setting of Train."""
    title = "Learning-rate schedules"
    state = "the step count"
    warmup: int
    total: int

    def init(self) -> int: ...

    def step(self, state: int, t: int, /) -> tuple[int, float]: ...


@dataclass(frozen=True)
class Lagrangian(Scan[tuple[Tensor, Tensor], float, Tensor]):
    """(loss f, constraint g) in, f + λ·(g − c) out; λ rises while the constraint is violated (dual ascent)."""
    title = "Constrained objectives (Lagrangian)"
    state = "the multiplier λ ≥ 0"
    lr: float = 0.1

    def init(self) -> float: ...

    def step(self, state: float, losses: tuple[Tensor, Tensor], /) -> tuple[float, Tensor]: ...


@dataclass(frozen=True)
class ReturnsToGo(Scan[Tensor, Tensor, Tensor]):
    """G_t = r_t + γ·G_{t+1}: what each step led to, scanned from the end."""
    title = "Returns-to-go"
    state = "the return from here on"
    reverse = True
    gamma: float = 1.0

    def init(self) -> Tensor: ...

    def step(self, state: Tensor, reward: Tensor, /) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class Negamax(Scan[Tensor, Tensor, Tensor]):
    """One policy plays both sides; each move's return is the outcome for whoever made it: G_t = r_t − G_{t+1}."""
    title = "Self-play"
    state = "the outcome for the player to move"
    reverse = True

    def init(self) -> Tensor: ...

    def step(self, state: Tensor, reward: Tensor, /) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class GAE(Scan[tuple[Tensor, Tensor], Tensor, Tensor]):
    """A critic's TD errors r + γV(s′) − V(s), averaged backward with weights (γλ)^k; each element a (reward, value)."""
    title = "Value functions, Bellman, GAE"
    state = "the advantage from here on"
    reverse = True
    gamma: float = 0.99
    lmbda: float = 0.95

    def init(self) -> Tensor: ...

    def step(self, state: Tensor, reward_value: tuple[Tensor, Tensor], /) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class ParallelScan(Scan[Tensor, Tensor, Tensor]):
    """s_t = a_t·s_{t−1} + b_t for every t at once, in logarithmic depth: how a recurrence trains in parallel."""
    title = "Parallel scan"
    function = "torch.cumsum"


# ═══ Split[C, **R = []]: C → Example ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AutoSplit(Split[Container]):
    """The split a container's type picks: classes in order → NextToken · other classes → Mask · real values → Noise."""
    title = "Choosing the split from the data"

    def __call__(self, x: Container, /) -> Example: ...


@dataclass(frozen=True)
class Supervised(Split[Container, [Container]]):
    """The data comes split: the input with nothing hidden, the label as the target."""
    title = "Supervised"

    def __call__(self, x: Container, y: Container, /) -> Example: ...


@dataclass(frozen=True)
class Classification(Split[Container, [Container]]):
    """Supervised learning with classes as the target: Supervised, then Loss picks cross-entropy (a categorical's
    negative log-likelihood); Accuracy measures it. Any model that predicts the classes does it: Linear (logistic
    regression), Network, KNN, a boosted-trees estimator."""
    title = "Classification"
    function = "Supervised, Loss"


@dataclass(frozen=True)
class Regression(Split[Container, [Container]]):
    """Supervised learning with real values as the target: Supervised, then Loss picks squared error (a Gaussian's
    negative log-likelihood). Linear with LeastSquares is linear regression; Network, KNN and boosted trees do it too."""
    title = "Regression"
    function = "Supervised, Loss"


@dataclass(frozen=True)
class NextToken(Split[Temporal]):
    """Each element is predicted from the ones before it (a causal Temporal): the target at i is the element at i + 1."""
    title = "Self-supervised: next-token prediction"

    def __call__(self, x: Temporal, /) -> Example: ...


@dataclass(frozen=True)
class Mask(Split[Container]):
    """Random elements at level 1 ([MASK] for classes, 0 for real values): 15% is BERT, 75% MAE, a random ratio per
    example MaskGIT and masked diffusion."""
    title = "Self-supervised: masked prediction (BERT, MAE, MaskGIT)"
    ratio: float | None = None

    def __call__(self, x: Container, /) -> Example: ...


@dataclass(frozen=True)
class Noise(Split[Container]):
    """Every element at one random level t (x_t = (1 − t)·x + t·ε): the target is the velocity, the noise or the
    clean data, rescalings of one another."""
    title = "Diffusion / score matching / flow matching"

    def __call__(self, x: Container, /) -> Example: ...


@dataclass(frozen=True)
class DiffusionForcing(Split[Temporal]):
    """A random level for each element: next-token prediction and diffusion are its two extremes."""
    title = "Diffusion forcing"

    def __call__(self, x: Temporal, /) -> Example: ...


@dataclass(frozen=True)
class NextFrame(Split[Temporal[Grid]]):
    """The frames up to t as the input, the frames from t + 1 as the target: shift, on the frames of a video."""
    title = "Self-supervised: future-frame prediction"

    def __call__(self, x: Temporal[Grid], /) -> Example: ...


@dataclass(frozen=True)
class Reconstruction(Split[Container]):
    """Nothing hidden; the target is the input itself, so the model's bottleneck is what makes it hard."""
    title = "Reconstruction (autoencoder)"

    def __call__(self, x: Container, /) -> Example: ...


@dataclass(frozen=True)
class Contrastive(Split[Container, [Callable[[Container], Container]]]):
    """Two random augmentations of the same examples, each the other's target (CLIP pairs an image with its caption)."""
    title = "Contrastive views"

    def __call__(self, x: Container, augment: Callable[[Container], Container], /) -> Example: ...


@dataclass(frozen=True)
class Distillation(Split[Container, [Predictor]]):
    """The target is a teacher's (softened) prediction on the same input."""
    title = "Knowledge distillation"

    def __call__(self, x: Container, teacher: Predictor, /) -> Example: ...


@dataclass(frozen=True)
class SelfDistillation(Split[Container, [Predictor, Callable[[Container], Container]]]):
    """One view as the input; the target is a slow (EMA) copy's embedding of another view: agree with an older self."""
    title = "Self-distillation (BYOL / DINO / JEPA)"

    def __call__(self, x: Container, teacher: Predictor, augment: Callable[[Container], Container], /) -> Example: ...


@dataclass(frozen=True)
class SFT(Split[Temporal]):
    """shift, with only the response as targets: the prompt is read, never predicted."""
    title = "Instruction tuning / SFT"
    prompt_len: int

    def __call__(self, x: Temporal, /) -> Example: ...


@dataclass(frozen=True)
class TD(Split[Value, [Predictor]]):
    """V(s) learns toward r + γ·V(s′): a target bootstrapped from the critic's own estimate."""
    title = "TD learning (value regression)"
    gamma: float = 0.99

    def __call__(self, batch: Value, critic: Predictor, /) -> Example: ...


@dataclass(frozen=True)
class QLearning(Split[Value, [Predictor]]):
    """Q(s, a) learns toward r + γ·max Q(s′, ·) at the action taken (the weight picks it out)."""
    title = "Q-learning (DQN)"
    gamma: float = 0.99

    def __call__(self, batch: Value, q: Predictor, /) -> Example: ...


@dataclass(frozen=True)
class Dynamics(Split[Value]):
    """From each transition, the input is the state and the action; the target what followed (next state, reward,
    done). A model of it is a learned follow (imagined)."""
    title = "World-model targets (dynamics)"

    def __call__(self, batch: Value, /) -> Example: ...


@dataclass(frozen=True)
class AlphaZeroTargets(Split[Path, [PartialTree[Any, Any]]]):
    """For each state a walk visited: what search made of it (its visit distribution) and what followed (the return)."""
    title = "Amortized search (AlphaZero targets)"

    def __call__(self, path: Path, search: PartialTree[Any, Any], /) -> Example: ...


# ═══ Score[P, O = Tensor, **R = []]: Prediction × C → ℝ ═══════════════════════════════════════════


@dataclass(frozen=True)
class Objective(Score[nn.Module, Tensor, ...]):
    """The loss of a model on a batch: split it into an Example, predict from the input, score with the loss the
    target picks (labelled data: split = Supervised())."""
    title = "Objective (a split, then its loss)"
    split: Split[Any, ...]

    def __call__(self, model: nn.Module, /, *batch: Any) -> Tensor: ...


@dataclass(frozen=True)
class Loss(Score[Prediction, Tensor, [Container]]):
    """−log p(target), chosen by what the target holds: classes → cross-entropy · bits → BCE · real → squared error ·
    distribution → KL · record → the sum of its fields' losses."""
    title = "Loss = negative log-likelihood of the target"

    def __call__(self, pred: Prediction, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class CrossEntropy(Score[Tensor, Tensor, [Container]]):
    """Classes, or a distribution over them (soft labels)."""
    title = "Cross-entropy"

    def __call__(self, logits: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class BinaryCrossEntropy(Score[Tensor, Tensor, [Container]]):
    """Bits."""
    title = "Binary cross-entropy"

    def __call__(self, logits: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class Hinge(Score[Tensor, Tensor, [Value]]):
    """max(0, margin − y·f(x)) for labels ±1 (for classes: the margin over the best wrong class): a support vector
    machine is a Linear trained with it (on RandomFourierFeatures for a kernel SVM)."""
    title = "Hinge loss (SVM)"
    margin: float = 1.0

    def __call__(self, scores: Tensor, target: Value, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class MSE(Score[Tensor, Tensor, [Container]]):
    """Real values: a unit Gaussian's negative log-likelihood, up to a constant."""
    title = "Squared error"

    def __call__(self, pred: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class KL(Score[Dist, Tensor, [Dist]]):
    """A distribution against a reference (a prior, a reference policy)."""
    title = "KL divergence"

    def __call__(self, pred: Dist, target: Dist, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class MultiTaskLoss(Score[Prediction, Tensor, [Container]]):
    """A record: one loss per field, each chosen by what the field holds, summed."""
    title = "Multi-task training"

    def __call__(self, pred: Prediction, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class LogProb(Score[Prediction, Value, [Container]]):
    """log p(x) for each element under a prediction, unreduced: what losses and estimators are built from."""
    title = "Log-probability"

    def __call__(self, pred: Prediction, x: Container, /) -> Value: ...


@dataclass(frozen=True)
class PathLogProb(Score[PartialTree[Any, Any], Value, [Path]]):
    """Σ log options(node)(edge) along a path; on the data's own path, −1/length of it is the next-token loss."""
    title = "Teacher forcing (the log-probability of a path)"

    def __call__(self, tree: PartialTree[Any, Any], path: Path, /) -> Value: ...


@dataclass(frozen=True)
class SequenceLogProb(Score[Predictor, Value, [Temporal]]):
    """log p(x[start:] | the rest before it), per sequence: teacher forcing on whole sequences, in one pass."""
    title = "Sequence log-probability"
    start: int = 1

    def __call__(self, model: Predictor, x: Temporal, /) -> Value: ...


@dataclass(frozen=True)
class InfoNCE(Score[Value, Tensor, [Value]]):
    """Which b goes with each a, among the batch: cross-entropy over similarities, the matching pair as the class."""
    title = "Contrastive (InfoNCE / CLIP)"
    temperature: float = 0.07

    def __call__(self, a: Value, b: Value, /) -> Tensor: ...


@dataclass(frozen=True)
class BradleyTerry(Score[Value, Tensor, [Value]]):
    """−log σ(r(chosen) − r(rejected)): the likelihood that the preferred answer wins."""
    title = "Reward models (Bradley–Terry)"

    def __call__(self, chosen: Value, rejected: Value, /) -> Tensor: ...


@dataclass(frozen=True)
class DPO(Score[nn.Module, Tensor, ...]):
    """The loss of (policy, chosen, rejected): Bradley–Terry on β·(the policy's log-ratio to the reference, chosen
    minus rejected). Preferences as classification: no reward model, no walk."""
    title = "Preferences (DPO)"
    reference: Predictor
    beta: float = 0.1

    def __call__(self, model: nn.Module, /, *batch: Any) -> Tensor: ...


@dataclass(frozen=True)
class KLPenalty(Score[Value, Value, [Value, Value]]):
    """The reward, minus how far the policy has moved from a reference: use it as the return."""
    title = "KL-regularized reward (RLHF)"
    beta: float = 0.1

    def __call__(self, reward: Value, LogProb: Value, ref_log_prob: Value, /) -> Value: ...


@dataclass(frozen=True)
class ELBO(Score[Model[Container, Dist], Tensor, [Model[Tensor, Prediction], Container]]):
    """The reconstruction from a latent drawn from the posterior (rsample: reparameterization), plus KL(posterior ‖ prior)."""
    title = "VAE (evidence lower bound)"

    def __call__(self, encoder: Model[Container, Dist], decoder: Model[Tensor, Prediction], x: Container, /) -> Tensor: ...


@dataclass(frozen=True)
class GAN(Score[Model[Tensor, Container], tuple[Tensor, Tensor], [Predictor, Container, Tensor]]):
    """The discriminator tells real from generated; the generator wants its samples called real. Two scores, for two
    players (alternate their steps)."""
    title = "GAN (two players)"

    def __call__(self, generator: Model[Tensor, Container], discriminator: Predictor, real: Container, z: Tensor, /) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class FlowLogProb(Score[Container, Value, [list[Any]]]):
    """The exact log-likelihood through invertible maps (change of variables)."""
    title = "Normalizing flows"
    side = "model"

    def __call__(self, x: Container, transforms: list[Any], /) -> Value: ...


@dataclass(frozen=True)
class Reinforce(Score[Path, Tensor, [Value]]):
    """−E[log p(path)·A]: a walked path's log-probability, weighted by how much better than usual it went."""
    title = "Policy gradients (REINFORCE)"

    def __call__(self, path: Path, advantage: Value, /) -> Tensor: ...


@dataclass(frozen=True)
class PPO(Score[Value, Tensor, [Value, Value]]):
    """The policy gradient with the ratio π / π_old clipped: it stops pushing once the policy has moved far."""
    title = "PPO clipped surrogate"
    clip: float = 0.2

    def __call__(self, LogProb: Value, old_log_prob: Value, advantage: Value, /) -> Tensor: ...


@dataclass(frozen=True)
class Pathwise(Score[Dist, Tensor, [Model[Tensor, Tensor]]]):
    """−E[Q(a)] with a = rsample(): differentiated through the sample into a learned critic (DDPG, SAC)."""
    title = "Pathwise estimator (through a critic)"

    def __call__(self, policy: Dist, critic: Model[Tensor, Tensor], /) -> Tensor: ...


@dataclass(frozen=True)
class EntropyBonus(Score[Dist]):
    """−α·H(options): keeps the policy exploring."""
    title = "Maximum-entropy RL (entropy bonus)"
    alpha: float = 0.01

    def __call__(self, options: Dist, /) -> Tensor: ...


@dataclass(frozen=True)
class MAML(Score[nn.Module, Tensor, [Callable[..., Tensor], Container, Container]]):
    """Bilevel: the query loss of the weights after an inner step (an unroll) on the support set, differentiated through."""
    title = "Meta-learning (MAML)"

    def __call__(self, model: nn.Module, objective: Callable[..., Tensor], support: Container, query: Container, /) -> Tensor: ...


@dataclass(frozen=True)
class Accuracy(Score[Tensor, Tensor, [Container]]):
    """Reported and selected on, never differentiated: a loss stands in for it in training."""
    title = "Metrics (accuracy)"

    def __call__(self, pred: Tensor, target: Container, /) -> Tensor: ...


@dataclass(frozen=True)
class RetrievalAccuracy(Score[Value, Tensor, [Value]]):
    """How often each a is closest to its own b among the batch (matching images to captions)."""
    title = "Retrieval accuracy"

    def __call__(self, a: Value, b: Value, /) -> Tensor: ...


@dataclass(frozen=True)
class Grad(Score[Tensor, Tensor]):
    """The gradient of any score."""
    title = "Backpropagation / autodiff"
    function, signature = "torch.autograd", "ℝ → ∇"


# ═══ Start[P, **R = []]: … → Partial ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Blank(Start[Holes[Container], [Container]]):
    """Every element unknown: all [MASK] for classes, pure noise for real values."""
    title = "Blank (where generation starts)"

    def __call__[C: Container](self, like: C, /) -> Holes[C]: ...


@dataclass(frozen=True)
class LMTree(Start[PartialTree[Temporal, Tensor], [Predictor, Temporal]]):
    """A node is a sequence (Holes: known up to its end), its options the next-token distribution, an edge appends."""
    title = "An LM as a tree"

    def __call__(self, model: Predictor, prompt: Temporal, /) -> PartialTree[Temporal, Tensor]: ...


@dataclass(frozen=True)
class EnvTree(Start[PartialTree[Any, Tensor], [Environment]]):
    """A node is a state, its options the policy's actions (uniform with none), an edge leads where the environment
    steps to, with a reward and whether it ended."""
    title = "MDP (an environment as a tree)"

    def __call__(self, env: Environment, policy: Predictor | None = None, /) -> PartialTree[Any, Tensor]: ...


# ═══ Order: Holes × Prediction → Holes ════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LeftToRight(Order):
    """Fill the first hole: one element at a time, in order (the holes view of decoding; walk gives the tree view)."""
    title = "Autoregressive order"
    state = "Holes[C] — known up to i"

    def __call__(self, holes: Holes[Any], pred: Prediction, /) -> Holes[Any]: ...


@dataclass(frozen=True)
class MostConfident(Order):
    """Fill the holes the model is surest of, more each step (a cosine schedule), until none is left."""
    title = "Unmasking (MaskGIT, masked diffusion)"
    state = "Holes[C] — some holes at 1, the rest known"

    def __call__(self, holes: Holes[Any], pred: Prediction, /) -> Holes[Any]: ...


@dataclass(frozen=True)
class AllAtOnce(Order):
    """Lower every level by one step along the predicted velocity (Euler): noise, undone."""
    title = "Sampling a flow (diffusion)"
    state = "Holes[C] — every element at level t"

    def __call__(self, holes: Holes[Any], pred: Prediction, /) -> Holes[Any]: ...


@dataclass(frozen=True)
class PerElement(Order):
    """Each element follows its own schedule (the near future clean, the far future noisy): between LeftToRight and together."""
    title = "Diffusion forcing sampling"
    state = "Holes[C] — a level for each element"

    def __call__(self, holes: Holes[Any], pred: Prediction, /) -> Holes[Any]: ...


# ═══ Walk: PartialTree → Path ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChainOfThought(Walk):
    """A longer walk before the answer; the gains come from training the walk (R1)."""
    title = "Chain-of-thought / longer reasoning"
    function = "Walk"


# ═══ Choose: PartialTree → PartialTree ════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Sample(Choose):
    """Expand each kept leaf by one edge drawn from its options."""
    title = "Sampling"
    state = "PartialTree — one leaf per root"

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class Greedy(Choose):
    """Expand each kept leaf by its likeliest edge."""
    title = "Greedy decoding"
    state = "PartialTree — one leaf per root"

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class Beam(Choose):
    """Expand every kept leaf; keep the k likeliest children of each root."""
    title = "Beam search"
    state = "k leaves per root and their log-probabilities"
    k: int

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class NPaths(Choose):
    """n paths from each root side by side, each grown by its own chooser: n samples per prompt (best-of-N, GRPO)."""
    title = "Several paths per root (best-of-N, groups)"
    state = "n independent leaves per root"
    n: int
    choose: Chooser

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class TeacherForcing(Choose):
    """Expand each leaf by the next given edge: scoring the path walked is teacher forcing."""
    title = "Teacher forcing (walking the data's path)"
    state = "one leaf per root, on the data's path"
    edges: Seq

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class Speculative(Choose):
    """The draft walks k edges ahead greedily, the tree's model checks them in one pass, the agreeing prefix is kept:
    greedy decoding, in fewer of the big model's passes."""
    title = "Speculative decoding"
    state = "the accepted leaf, and a draft model's guesses past it"
    draft: PartialTree[Any, Any]
    k: int = 4

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


# ═══ Unroll: PartialSeq[S] → S ════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Iteration(Unroll):
    """Rounds of collecting and learning (policy iteration, self-play, expert iteration): a script's repeat
    (program.py), since each round's fit is a statement."""
    title = "Iteration (rounds of a program)"
    function = "repeat"


# ═══ Evolve[S]: S → S ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Langevin(Evolve[Container]):
    """x ← x − η·∇E(x) + √(2η)·ε: samples of exp(−energy), with the gradient from autograd."""
    title = "Energy-based models (Langevin)"
    state = "a sample, moving"
    energy: Predictor
    step_size: float = 0.01

    def __call__(self, state: Container, /) -> Container: ...


@dataclass(frozen=True)
class EvolutionStrategy(Evolve[Dist]):
    """Draw a population (a Set of parameters) from the search distribution, score them, move it toward the better
    ones: it works when nothing is differentiable (CMA-ES keeps a covariance too)."""
    title = "Evolution strategies (no gradient)"
    state = "a search distribution over the parameters"
    fitness: Callable[[Tensor], Tensor]
    population: int = 32

    def __call__(self, state: Dist, /) -> Dist: ...


# ═══ Reshape[P, **R = []]: Partial → Partial ══════════════════════════════════════════════════════


@dataclass(frozen=True)
class Temperature(Reshape[PartialTree[Any, Any]]):
    """The options' logits divided by t: below 1 sharper (toward greedy), above 1 flatter."""
    title = "Sampling temperature"
    t: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class TopK(Reshape[PartialTree[Any, Any]]):
    """Only the k likeliest edges stay possible."""
    title = "Top-k sampling"
    k: int

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class TopP(Reshape[PartialTree[Any, Any]]):
    """Only the likeliest edges whose probabilities add up to p stay possible."""
    title = "Top-p (nucleus) sampling"
    p: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class Given(Reshape[Predictor, [Value | None]]):
    """A conditional model with its condition fixed (None: no condition): the two models guidance mixes."""
    title = "A condition, fixed"

    def __call__(self, model: Predictor, condition: Value | None, /) -> Predictor: ...


@dataclass(frozen=True)
class CFG(Reshape[Predictor, [Predictor]]):
    """u + w·(c − u): extrapolate from the unconditional prediction toward the conditional one, for a tree's options or
    a fill's model."""
    title = "Classifier-free guidance"
    w: float

    def __call__(self, cond: Predictor, uncond: Predictor, /) -> Predictor: ...


@dataclass(frozen=True)
class EpsilonGreedy(Reshape[PartialTree[Any, Any]]):
    """With probability ε a uniform edge, else the likeliest."""
    title = "Exploration vs exploitation (ε-greedy)"
    eps: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class ToolUse(Reshape[PartialTree[Any, Any], [Callable[[Any], Tensor]]]):
    """When a walk emits the call token, follow writes the tool's answer (a calculator, a search, an interpreter)."""
    title = "Tool use"
    call: int

    def __call__(self, tree: PartialTree[Any, Any], tool: Callable[[Any], Tensor], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class KVCache(Reshape[PartialTree[Any, Any]]):
    """The tree's known nodes keep what the model computed for them, so expanding a node costs one position."""
    title = "KV caching"
    state = "each known node keeps its keys and values"

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class WorldModel(Reshape[PartialTree[Any, Any], [Model[Container, Container]]]):
    """The environment's follow (and reward, done) replaced by a learned model of them: walks, searches and plans in
    imagination (Dreamer, MuZero)."""
    title = "World models (a learned follow)"

    def __call__(self, tree: PartialTree[Any, Any], world: Model[Container, Container], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class MPC(Reshape[PartialTree[Any, Any]]):
    """Options that look ahead: random action sequences simulated in a world, all the probability on the first action
    of the best."""
    title = "Planning (random shooting / MPC)"
    state = "the random plans tried from each node"
    horizon: int = 5
    samples: int = 64

    def __call__(self, tree: PartialTree[Any, Any], world: PartialTree[Any, Any] | None = None, /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class MCTS(Reshape[PartialTree[Any, Any]]):
    """Options improved by search: grow a tree by PUCT (the options as priors), back up rewards and values, return the
    visit counts. Walk it to search; train toward it to amortize the search (AlphaZero)."""
    title = "MCTS (tree search)"
    state = "a search tree per node: priors, visit counts, values"
    simulations: int = 64
    two_player: bool = False

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


@dataclass(frozen=True)
class CFGDropout(Reshape[Predictor]):
    """The condition dropped with probability p while training, so one model is both conditional and unconditional."""
    title = "Classifier-free guidance training"
    p: float = 0.1

    def __call__(self, model: Predictor, /) -> Predictor: ...


@dataclass(frozen=True)
class KNNLM(Reshape[PartialTree[Any, Any], [Index, Predictor]]):
    """Options mixed with the next tokens of the k nearest stored contexts: an LM that can look things up."""
    title = "kNN-LM (a memory of contexts)"
    weight: float = 0.25

    def __call__(self, tree: PartialTree[Any, Any], memory: Index, embed: Predictor, /) -> PartialTree[Any, Any]: ...


# ═══ Search[O, **R = []]: Index × Value → Set ═════════════════════════════════════════════════════


@dataclass(frozen=True)
class Retrieve(Search[Temporal, [Set, Temporal, Predictor]]):
    """The k documents nearest the query's embedding, put in front of it as one prompt (then walk the LM's tree)."""
    title = "Retrieval-augmented generation (retrieve)"
    k: int = 4

    def __call__(self, index: Index, documents: Set, query: Temporal, embed: Predictor, /) -> Temporal: ...


@dataclass(frozen=True)
class HardNegatives(Search[Set, [Value, Value]]):
    """The nearest elements that are not the match: harder negatives for a contrastive loss."""
    title = "Hard-negative mining"
    k: int = 8

    def __call__(self, index: Index, queries: Value, positives: Value, /) -> Set: ...


@dataclass(frozen=True)
class KNN(Search[Prediction, [Value, Value]]):
    """k-nearest-neighbour prediction: each query's k nearest stored examples (index.search), their labels voted
    (classes) or averaged (real values), weighted by distance. Nothing to fit: Insert the examples, then ask."""
    title = "k-nearest neighbours (classification, regression)"
    k: int = 5

    def __call__(self, index: Index, queries: Value, labels: Value, /) -> Prediction: ...


# ═══ Wrap[M, **R = []]: Module → Module′ ══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Observe(Wrap[nn.Module, [nn.Module]]):
    """A model that returns what one of its modules computed: the activations probes, SAEs and circuits study."""
    title = "Reading activations"

    def __call__(self, model: nn.Module, at: nn.Module, /) -> nn.Module: ...


@dataclass(frozen=True)
class Patch(Wrap[nn.Module, [nn.Module, Callable[[Tensor], Tensor]]]):
    """A model with one module's output replaced (by another input's activations): which components carry a behaviour?"""
    title = "Activation patching (circuits)"

    def __call__(self, model: nn.Module, at: nn.Module, replace: Callable[[Tensor], Tensor], /) -> nn.Module: ...


@dataclass(frozen=True)
class Steer(Wrap[nn.Module, [nn.Module, Tensor]]):
    """α·direction added to one module's output: features are directions, and moving along one changes the behaviour."""
    title = "Steering along a direction"
    alpha: float = 1.0

    def __call__(self, model: nn.Module, at: nn.Module, direction: Tensor, /) -> nn.Module: ...


@dataclass(frozen=True)
class Autocast(Wrap[nn.Module]):
    """bf16 (or fp8) compute, float32 master weights."""
    title = "Low precision: training"
    function = "torch.autocast"


@dataclass(frozen=True)
class Compile(Wrap[nn.Module]):
    """torch.compile(module) fuses its ops; attention picks flash kernels by itself."""
    title = "Hardware-aware kernels"
    side = "model"
    function = "torch.compile"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# STATEMENTS · making new state, changing it in place (!), or reading a whole dataset
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ═══ MapLayer[A, B, **R = []]: builds C[a] → C[b] ═════════════════════════════════════════════════


@dataclass(frozen=True)
class Network(MapLayer[Container, Prediction, [Container, Container]]):
    """embed → position → [aggregate, MLP] × depth → pool → readout, each slot chosen from the input and target types
    (Embed, AutoAggregate, Pool, Readout); mixer and mlp replace a slot's choice with a config (a scan,
    experts). Calling it builds the network for inputs like x and targets like y."""
    title = "Network (each layer chosen by the types)"
    d: int = 64
    depth: int = 2
    mixer: AggregateLayer[Any] | None = None
    mlp: MapLayer[Any, Any] | None = None

    def __call__(self, x: Container, y: Container, /) -> Predictor: ...


@dataclass(frozen=True)
class Encoder(MapLayer[Container, Value, [Container]]):
    """A network from any container to one vector per example: the towers of CLIP, a reward model, a critic."""
    title = "Embedding models (a container to one vector)"
    dim: int

    def __call__(self, x: Container, /) -> Model[Container, Value]: ...


@dataclass(frozen=True)
class Embed(MapLayer[Container, Container, [Container]]):
    """By what the elements hold: classes → a row of a lookup table (one more for [MASK]); real values → a linear map."""
    title = "Embeddings"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class Readout(MapLayer[Container, Container, [Container]]):
    """By what the target holds: a logit per class, a value per real number, one readout per field of a record."""
    title = "Readout"
    d: int

    def __call__(self, y: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class LearnedPositions(MapLayer[Container, Container, [Container]]):
    """A learned vector added at each position of every Seq level (Temporal, Spatial: a Grid has two); a Set, a Graph
    or a Tree gets none."""
    title = "Positional encoding: learned"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class SinusoidalPositions(MapLayer[Container, Container, [Container]]):
    """Fixed sin/cos features of each position at geometrically spaced frequencies, added."""
    title = "Positional encoding: sinusoidal"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class MoE(MapLayer[Container, Container]):
    """A router sends each element to its top-k experts (MLPs) and sums their gate-weighted outputs."""
    title = "Mixture of experts"
    d: int
    experts: int = 8
    k: int = 2

    def __call__(self) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class Autoencoder(MapLayer[Container, Container, [Container]]):
    """encode to a smaller container (fewer positions, few channels), decode back: the latent must keep what matters,
    and generation can happen there (latent diffusion). Its .encode and .decode are the two halves."""
    title = "Latent space / bottleneck"
    latent: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class SAE(MapLayer[Container, Container]):
    """Activations → a wide, sparse code → activations, trained by reconstruction plus an L1 penalty: each code unit
    tends to be one feature."""
    title = "Sparse autoencoders (superposition)"
    d: int
    features: int

    def __call__(self) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class LinearProbe(MapLayer[Value, Value, [Value, Value]]):
    """A linear readout for frozen activations, sized by examples of them and their labels; fitted by a statement
    (LeastSquares, Train). Is a feature readable along one direction?"""
    title = "Linear probes (features as directions)"

    def __call__(self, acts: Value, labels: Value, /) -> Model[Value, Value]: ...


@dataclass(frozen=True)
class MLP(MapLayer[Container, Container]):
    """The same small network on every element."""
    title = "Linear layer + nonlinearity (MLP)"
    function, signature = "nn.Linear, nn.GELU, nn.Sequential", "C[h] → C[h]"


@dataclass(frozen=True)
class SwiGLU(MapLayer[Container, Container]):
    """silu(gate(h)) · up(h), then down: one branch decides how much of the other passes."""
    title = "Gating (SwiGLU)"
    function, signature = "nn.SiLU", "C[h] → C[h]"


@dataclass(frozen=True)
class Norm(MapLayer[Container, Container]):
    """Rescale each element's features (BatchNorm aggregates over the batch instead)."""
    title = "Normalization"
    function, signature = "nn.RMSNorm, nn.LayerNorm", "C[h] → C[h]"


@dataclass(frozen=True)
class Dropout(MapLayer[Container, Container]):
    """Zero random features while training."""
    title = "Dropout"
    function, signature = "nn.Dropout", "C[h] → C[h]"


@dataclass(frozen=True)
class Linear(MapLayer[Value, Prediction, [Value, Container]]):
    """A linear model Wx + b on feature vectors, sized by examples of its inputs and targets: linear regression when
    the target is real values (Loss picks squared error), logistic or softmax regression when it is classes
    (cross-entropy), a linear SVM with Hinge. Fitted by Train (a gradient), or LeastSquares (closed form)."""
    title = "Linear models (linear and logistic regression)"
    bias: bool = True

    def __call__(self, x: Value, y: Container, /) -> Model[Value, Prediction]: ...


@dataclass(frozen=True)
class RandomFourierFeatures(MapLayer[Value, Value, [Value]]):
    """Random Fourier features of a shift-invariant kernel (RBF by default): a Linear on them is a kernel machine, so
    kernel ridge regression is LeastSquares on it and a kernel SVM is Hinge on it."""
    title = "Kernel methods (random features)"
    features: int = 1024
    bandwidth: float = 1.0

    def __call__(self, x: Value, /) -> Model[Value, Value]: ...


@dataclass(frozen=True)
class Residual(MapLayer[Container, Container, [nn.Module]]):
    """h + f(norm h): the layer edits the stream instead of replacing it."""
    title = "Residual connections"
    side = "model"

    def __call__(self, f: nn.Module, /) -> Model[Container, Container]: ...


# ═══ AggregateLayer[C, In = C, **R = []]: builds C[h] → C[h] ══════════════════════════════════════


@dataclass(frozen=True)
class AutoAggregate(AggregateLayer[Container, Container, [Container]]):
    """What each element reads, level by level of the container: Graph, Tree → message passing · causal Temporal →
    causal attention (a selective scan when long) · Spatial (a large Grid) → convolution · several levels → axial ·
    else attention over all."""
    title = "Choosing the aggregation from the container"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class Attention(AggregateLayer[Container]):
    """Each element reads every other (or every earlier one), weighted by query·key similarity."""
    title = "Attention (self, causal)"
    d: int
    causal: bool = False

    def __call__(self) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class CrossAttention(AggregateLayer[Container, tuple[Container, Container]]):
    """Queries from one container, keys and values from another (text reading an image)."""
    title = "Cross-attention"
    d: int

    def __call__(self) -> Model[tuple[Container, Container], Container]: ...


@dataclass(frozen=True)
class Conv(AggregateLayer[Grid]):
    """Each element reads its neighbours on the lattice, with the same weights at every offset."""
    title = "Convolution"
    d: int
    size: int = 3

    def __call__(self) -> Model[Grid, Grid]: ...


@dataclass(frozen=True)
class RNN(AggregateLayer[Temporal]):
    """Each element reads a state carried from the earlier ones, updated by a gated nonlinear function."""
    title = "Recurrence (RNN / LSTM / GRU)"
    state = "a hidden vector, carried along the sequence"
    d: int

    def __call__(self) -> Model[Temporal, Temporal]: ...


@dataclass(frozen=True)
class SSM(AggregateLayer[Seq]):
    """A linear recurrence whose coefficients come from the input (S4, Mamba, linear attention): every earlier element
    read, in linear time."""
    title = "State-space models / linear recurrences"
    state = "s_t = a_t·s_{t−1} + b_t, run as a parallel scan"
    d: int

    def __call__(self) -> Model[Seq, Seq]: ...


@dataclass(frozen=True)
class GNN(AggregateLayer[Graph]):
    """Each node reads its own features and the mean along its incoming edges; in a tree, children pass up to parents."""
    title = "Message passing (GNN)"
    d: int

    def __call__(self) -> Model[Graph, Graph]: ...


@dataclass(frozen=True)
class Axial(AggregateLayer[Container]):
    """One aggregation per level of a nested container (a video's frames, then its rows, then its columns), the other
    levels as batch; summed."""
    title = "Axial (nested structures)"
    d: int

    def __call__(self) -> Model[Container, Container]: ...


@dataclass(frozen=True)
class UNet(AggregateLayer[Grid]):
    """Down to a coarse grid and back up, with skip connections: each element reads a wide neighbourhood."""
    title = "Encoder–decoder hierarchy (U-Net)"
    d: int
    levels: int = 2

    def __call__(self) -> Model[Grid, Grid]: ...


@dataclass(frozen=True)
class RetrievalAttention(AggregateLayer[Container, tuple[Container, Index]]):
    """Each element reads the k nearest entries of an external memory (an Index), softmax-weighted."""
    title = "Retrieval (attention over a memory)"
    k: int

    def __call__(self) -> Model[tuple[Container, Index], Container]: ...


# ═══ FoldLayer[C, A, **R = []]: builds C[a] → a ═══════════════════════════════════════════════════


@dataclass(frozen=True)
class Pool(FoldLayer[Container, Value, [Container]]):
    """A container's features → one vector per example: a causal Temporal keeps its last position, the one that read
    everything; anything else, the mean."""
    title = "Pooling (mean / last position)"

    def __call__(self, x: Container, /) -> Model[Container, Value]: ...


# ═══ Encode[A, B, **R = []]: builds A ⇄ C[classes] ════════════════════════════════════════════════


@dataclass(frozen=True)
class Tokenizer(Encode[list[str], Temporal]):
    """Text ⇄ a sequence of ids, built empty: BPE merges, fitted to a corpus by FitCodec (bytes alone: a fixed
    vocabulary of 256, nothing to fit). The tokenizer decides the content type: Temporal of classes(vocab)."""
    title = "Tokenization (BPE, SentencePiece, bytes)"
    kind: str = "bpe"
    vocab: int = 32000

    def __call__(self) -> Codec[list[str], Temporal]: ...


@dataclass(frozen=True)
class ImageTokenizer(Encode[Grid, Grid, [Grid]]):
    """An image ⇄ a coarser grid of codebook ids: an encoder, vector quantization, a decoder, trained to reconstruct."""
    title = "Image tokenizer (VQ-VAE, VQGAN)"
    codes: int = 1024

    def __call__(self, images: Grid, /) -> Codec[Grid, Grid]: ...


@dataclass(frozen=True)
class AudioCodec(Encode[Temporal, Temporal[Seq], [Temporal]]):
    """A waveform ⇄ frames of codebook ids (a Temporal of Seqs): each codebook quantizes what the previous ones left
    (residual VQ)."""
    title = "Audio codec (EnCodec, residual quantization)"
    codebooks: int = 8
    codes: int = 1024

    def __call__(self, audio: Temporal, /) -> Codec[Temporal, Temporal[Seq]]: ...


# ═══ Empty[S, **R = []]: … → Index | Buffer ═══════════════════════════════════════════════════════


@dataclass(frozen=True)
class NewIndex(Empty[Index, [Value]]):
    """An empty index for vectors like these (their size): flat (exact), IVF (clusters), HNSW (a neighbour graph), PQ
    (compressed codes), LSH (hash buckets), inverted (BM25), sum tree (by priority). Statements train and fill it."""
    title = "Index (an empty one)"
    kind: str = "flat"

    def __call__(self, like: Value, /) -> Index: ...


@dataclass(frozen=True)
class ReplayBuffer(Empty[Buffer[Any]]):
    """An empty buffer of at most size examples, read by sampling (uniformly, or by priority); Extend fills it."""
    title = "Replay buffer (empty)"
    size: int
    priority: bool = False

    def __call__(self) -> Buffer[Any]: ...


# ═══ Copy[M]: Module → Module ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FrozenCopy(Copy[nn.Module]):
    """A copy that no longer trains: DPO's and RLHF's reference, DQN's target network."""
    title = "Frozen copy (reference / target network)"

    def __call__[M: nn.Module](self, model: M, /) -> M: ...


# ═══ Sklearn[E]: Estimator → SkModel ══════════════════════════════════════════════════════════════
# The classical models, by the estimators that make them: each is Sklearn with a name, Sklearn(estimator) all the same.


@dataclass(frozen=True)
class Trees(Sklearn[Estimator]):
    """Decision trees on feature vectors: one (CART), many on bootstrap samples averaged (a random forest), or each
    fitted to the errors of the ones before it (gradient boosting). Classifier or regressor, by the estimator."""
    title = "Decision trees, random forests, gradient boosting"
    function = "sklearn.ensemble.HistGradientBoostingClassifier / Regressor, RandomForestClassifier, sklearn.tree.DecisionTreeClassifier, xgboost, lightgbm"


@dataclass(frozen=True)
class SVM(Sklearn[Estimator]):
    """The widest-margin separator in a kernel's feature space (a linear one is Linear with Hinge)."""
    title = "Support vector machines (kernel SVM)"
    function = "sklearn.svm.SVC, SVR"


@dataclass(frozen=True)
class NaiveBayes(Sklearn[Estimator]):
    """A generative classifier whose features are independent given the class: counts or one Gaussian per feature."""
    title = "Naive Bayes"
    function = "sklearn.naive_bayes.GaussianNB, MultinomialNB"


@dataclass(frozen=True)
class GaussianProcess(Sklearn[Estimator]):
    """A distribution over functions, set by a kernel: a prediction and its uncertainty, from every training example."""
    title = "Gaussian processes"
    function = "sklearn.gaussian_process.GaussianProcessRegressor, GaussianProcessClassifier"


@dataclass(frozen=True)
class KMeans(Sklearn[Estimator]):
    """k centroids, each example to its nearest and each centroid to the mean of its examples, until nothing moves;
    predict gives a new example's cluster."""
    title = "k-means"
    function = "sklearn.cluster.KMeans, MiniBatchKMeans"


@dataclass(frozen=True)
class GMM(Sklearn[Estimator]):
    """k weighted Gaussians fitted by expectation–maximization: a density, and soft clusters by responsibility."""
    title = "Mixture models (GMM, EM)"
    function = "sklearn.mixture.GaussianMixture, BayesianGaussianMixture"


@dataclass(frozen=True)
class PCA(Sklearn[Estimator]):
    """A linear map onto the directions of most variance; transform projects onto them."""
    title = "PCA (dimensionality reduction)"
    function = "sklearn.decomposition.PCA, IncrementalPCA, TruncatedSVD"


@dataclass(frozen=True)
class Standardize(Sklearn[Estimator]):
    """Each feature shifted and scaled (to mean 0 and variance 1, or into a range), by statistics of the training data
    only, so that the validation and test data are scaled as it was."""
    title = "Standardization (feature scaling)"
    function = "sklearn.preprocessing.StandardScaler, MinMaxScaler, RobustScaler"


# ═══ Fit[X, **R = []]: X! × Dataset ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Train(Fit[nn.Module, [Callable[..., Tensor], Dataset[Any] | None]]):
    """Descend the objective over the dataset's batches: each step, objective(model, *batch) (under autocast to
    precision), its gradient (clipped to max_grad_norm), the optimizer's update of the parameters, in place, at the
    schedule's rate. In training mode only here, so a forward's own effects (BatchNorm statistics) are training's
    too. With no data, the objective makes its own each step (it walks the model: RL, self-play)."""
    title = "Training"
    steps: int
    optimizer: Scan[Tensor, Any, Tensor] = Optimizer()
    schedule: Scan[int, int, float] | None = None
    max_grad_norm: float | None = 1.0
    precision: torch.dtype | None = None

    def __call__(self, model: nn.Module, objective: Callable[..., Tensor], data: Dataset[Any] | None, /) -> None: ...


@dataclass(frozen=True)
class FitCodec(Fit[Codec[Any, Any], [Dataset[Any]]]):
    """Fit a codec to a corpus by its own rule, in place: BPE merges the most frequent pairs until the vocabulary is
    full, a unigram model prunes. (A learned codec, VQ-VAE or EnCodec, is a model: train.)"""
    title = "Codec fitting (BPE merges, a vocabulary)"

    def __call__(self, codec: Codec[Any, Any], data: Dataset[Any], /) -> None: ...


@dataclass(frozen=True)
class LeastSquares(Fit[nn.Module, [Dataset[Any]]]):
    """Solve for a Linear model's weights in closed form, in place: (XᵀX + λI)⁻¹Xᵀy, XᵀX and Xᵀy accumulated over
    the dataset's batches; ridge regression when ridge is above 0."""
    title = "Least squares (closed form, ridge)"
    side = "model"
    ridge: float = 0.0

    def __call__(self, model: nn.Module, data: Dataset[Any], /) -> None: ...


@dataclass(frozen=True)
class FitEstimator(Fit[SkModel[Any], [Dataset[Any]]]):
    """Fit an estimator's model to a dataset, in place: batch by batch by partial_fit when incremental (by default,
    whenever the estimator has it), otherwise every batch gathered and fit once. Each batch is (features, target), or
    features alone for an unsupervised one."""
    title = "Fitting an estimator (fit, partial_fit)"
    changes = 0
    incremental: bool | None = None

    def __call__(self, model: SkModel[Any], data: Dataset[Any], /) -> None: ...


@dataclass(frozen=True)
class TrainingStep(Fit[nn.Module]):
    """Train's settings: autocast the objective, clip the gradient norm."""
    title = "Training step (clipping, mixed precision)"
    function = "Train"


@dataclass(frozen=True)
class TwoPlayerGames(Fit[nn.Module]):
    """No single quantity goes down: each player's Train on its own loss, in turn, inside a repeat (GANs,
    adversarial training)."""
    title = "Two-player games (alternating fits)"
    function = "repeat, Train"


# ═══ Assign[X, **R = []]: Weights! ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Transfer(Assign[Weights, [Weights]]):
    """Start a model from a trained one: copy into the target the source's weights whose names and shapes match (the
    body), keep the target's head."""
    title = "Pretrain → finetune (transfer)"

    def __call__(self, target: Weights, source: Weights, /) -> None: ...


@dataclass(frozen=True)
class Init(Assign[Weights]):
    """Fan-in normal for matrices, zeros for output projections: each residual block starts as the identity."""
    title = "Initialization"

    def __call__(self, model: Weights, /) -> None: ...


@dataclass(frozen=True)
class EMA(Assign[Weights, [Weights]]):
    """average ← decay·average + (1 − decay)·model, in place: after each round or step, a smoother copy, and the slow
    target of self-distillation and DQN."""
    title = "EMA of weights"
    decay: float = 0.999

    def __call__(self, average: Weights, model: Weights, /) -> None: ...


# ═══ Alter[X, **R = []]: Module! ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Freeze(Alter[nn.Module, [Callable[[str], bool]]]):
    """Train only the parameters whose name passes a test."""
    title = "Freezing"
    side = "agnostic"

    def __call__(self, model: nn.Module, trainable: Callable[[str], bool], /) -> None: ...


@dataclass(frozen=True)
class LoRA(Alter[nn.Module]):
    """W + B·A on chosen linear layers, the base frozen: the finetuning update, kept low-rank."""
    title = "LoRA"
    rank: int = 8

    def __call__(self, model: nn.Module, /) -> None: ...


@dataclass(frozen=True)
class Adapter(Alter[nn.Module]):
    """After chosen frozen layers, a small bottleneck MLP that trains alone, starting at zero."""
    title = "Adapters"
    bottleneck: int = 8

    def __call__(self, model: nn.Module, /) -> None: ...


@dataclass(frozen=True)
class PromptTuning(Alter[nn.Module]):
    """Learned vectors in front of a frozen model's input; only they train."""
    title = "Prompt tuning"
    n: int

    def __call__(self, model: nn.Module, /) -> None: ...


@dataclass(frozen=True)
class Quantize(Alter[Weights]):
    """Weights rounded to 2^(bits−1) − 1 levels a side per row: computing as the low-precision model would."""
    title = "Quantization (int8 / int4)"
    bits: int = 8

    def __call__(self, model: Weights, /) -> None: ...


@dataclass(frozen=True)
class LowPrecisionInference(Alter[Weights]):
    """int8 / int4 weights."""
    title = "Low precision: inference"
    function = "Quantize"


@dataclass(frozen=True)
class Shard(Alter[nn.Module]):
    """Data, sharded, tensor, pipeline, expert, sequence: fully_shard shards a module in place; DTensor places its
    weights."""
    title = "Parallelism"
    side = "agnostic"
    function = "torch.distributed"


# ═══ Store[X, **R = []]: Index! / Buffer! ═════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrainIndex(Store[Index, [Value]]):
    """Learn the index's structure from a sample of the vectors, in place: k-means centroids for IVF, codebooks for
    PQ (flat and HNSW have nothing to learn). CREATE INDEX, before any rows."""
    title = "Index training (IVF centroids, PQ codebooks)"

    def __call__(self, index: Index, sample: Value, /) -> None: ...


@dataclass(frozen=True)
class Insert(Store[Index, [Dataset[Set]]]):
    """Add every batch of the dataset to the index, in place (FAISS add, INSERT). An index over an expression's output
    (embed(x)) goes stale when that expression's model is trained again: build it again."""
    title = "Index insertion"

    def __call__(self, index: Index, data: Dataset[Set], /) -> None: ...


@dataclass(frozen=True)
class Delete(Store[Index, [Tensor]]):
    """Remove elements from the index by id, in place (FAISS remove_ids, DELETE)."""
    title = "Index deletion"

    def __call__(self, index: Index, ids: Tensor, /) -> None: ...


@dataclass(frozen=True)
class Extend(Store[Buffer[Container], [Container]]):
    """Add a batch to a bounded buffer, in place, dropping the oldest (or the lowest priority): past experience to train
    on again (continual learning, off-policy RL, prioritized replay)."""
    title = "Continual learning (replay)"

    def __call__[C: Container](self, buffer: Buffer[C], batch: C, /) -> None: ...


# ═══ Source[D, **R = []]: … → Dataset ═════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Batches(Source[Indexed[Any], ...]):
    """Containers in memory as a dataset: random minibatches of size examples, the containers sliced together (a
    tuple per batch when there are several). Indexed, so it can be curated and split."""
    title = "Minibatches"
    size: int
    shuffle: bool = True

    def __call__(self, *data: Container) -> Indexed[Any]: ...


@dataclass(frozen=True)
class Collector(Source[Dataset[Path], [Environment, Predictor]]):
    """Walks of the environment with the policy as it is when each batch is drawn: on-policy data, read inside the
    fit that changes the policy (torchrl's collectors)."""
    title = "Collector (walks of a live policy)"
    steps: int
    batch: int

    def __call__(self, env: Environment, policy: Predictor, /) -> Dataset[Path]: ...


@dataclass(frozen=True)
class Mix(Source[Indexed[Any], ...]):
    """Several datasets as one (web, code, books), each example weighted so that batches draw from them in proportion
    to the weights: the mixture as one Indexed, to curate and split like any."""
    title = "Data mixture weights"
    weights: list[float]

    def __call__(self, *datasets: Indexed[Any]) -> Indexed[Any]: ...


@dataclass(frozen=True)
class MapBatches(Source[Dataset[Any], [Callable[..., Any], Dataset[Any]]]):
    """f applied to every batch, once per batch as the dataset batches it (never re-batched, since f may depend on
    what a batch holds): how a statement reads an expression's output, embeddings to index, features to probe."""
    title = "An expression on each batch"

    def __call__(self, f: Callable[..., Any], data: Dataset[Any], /) -> Dataset[Any]: ...


# ═══ Curate[C, **R = []]: Dataset → Dataset ═══════════════════════════════════════════════════════


@dataclass(frozen=True)
class Dedup(Curate[Any]):
    """Drop exact copies across the whole dataset (by hash): the first of each kept."""
    title = "Data deduplication"

    def indices(self, data: Indexed[Any], /) -> Tensor: ...


@dataclass(frozen=True)
class NearDedup(Curate[Any]):
    """Drop examples whose MinHash signatures collide with an earlier one's above a similarity threshold: dedup beyond
    exact copies."""
    title = "Near-duplicate removal (MinHash / LSH)"
    threshold: float = 0.8

    def indices(self, data: Indexed[Any], /) -> Tensor: ...


@dataclass(frozen=True)
class KeepBest(Curate[Any, [Callable[[Any], Value]]]):
    """The best-scoring fraction of the whole dataset: every score first, then the ones above their quantile."""
    title = "Data filtering by fraction"
    keep: float = 0.5

    def indices(self, data: Indexed[Any], score: Callable[[Any], Value], /) -> Tensor: ...


@dataclass(frozen=True)
class Curriculum(Curate[Any, [Callable[[Any], Value]]]):
    """The dataset sorted by difficulty, easy first: a global sort, so it reads every batch before the first comes out."""
    title = "Curriculum learning"

    def indices(self, data: Indexed[Any], difficulty: Callable[[Any], Value], /) -> Tensor: ...


@dataclass(frozen=True)
class Subsample(Curate[Any]):
    """A random fraction of the examples, without replacement: a smaller run, a quick look."""
    title = "Random subsampling"
    fraction: float
    seed: int = 0

    def indices(self, data: Indexed[Any], /) -> Tensor: ...


@dataclass(frozen=True)
class Resample(Curate[Any, [Callable[[Any], Value]]]):
    """n examples drawn with replacement, each in proportion to its weight: importance sampling, or the bootstrap when
    every weight is 1 (n defaults to the dataset's size)."""
    title = "Resampling (importance sampling, bootstrap)"
    n: int | None = None
    seed: int = 0

    def indices(self, data: Indexed[Any], weight: Callable[[Any], Value], /) -> Tensor: ...


@dataclass(frozen=True)
class Rebalance(Curate[Any, [Callable[[Any], Tensor]]]):
    """As many examples of each class as of the largest ("oversample", repeating the rarer ones) or the smallest
    ("undersample", dropping some of the commoner ones)."""
    title = "Class rebalancing (over- and undersampling)"
    mode: str = "oversample"
    seed: int = 0

    def indices(self, data: Indexed[Any], label: Callable[[Any], Tensor], /) -> Tensor: ...


# ═══ Partition[C, **R = []]: Indexed → splits of Subsets ══════════════════════════════════════════


@dataclass(frozen=True)
class TrainTestSplit(Partition[Any]):
    """One split, at random, into parts of the given fractions (train, validation, test), each class in the same
    proportions in every part when label says what an example's class is (stratified)."""
    title = "Train / validation / test split"
    fractions: tuple[float, ...] = (0.8, 0.1, 0.1)
    label: Callable[[Any], Tensor] | None = None
    seed: int = 0

    def parts(self, data: Indexed[Any], /) -> tuple[tuple[Tensor, ...], ...]: ...


@dataclass(frozen=True)
class Folds(Partition[Any]):
    """The splits of any scikit-learn splitter, each a (train, test) pair: label gives it the classes to stratify by,
    groups the groups to keep together (each called on batches, one value per example)."""
    title = "Splits from a splitter"
    splitter: Splitter
    label: Callable[[Any], Tensor] | None = None
    groups: Callable[[Any], Tensor] | None = None

    def parts(self, data: Indexed[Any], /) -> tuple[tuple[Tensor, ...], ...]: ...


@dataclass(frozen=True)
class KFold(Folds):
    """k (train, test) splits, each example tested once: plain, stratified by class, by group, or forward in time."""
    title = "k-fold cross-validation"
    function = "sklearn.model_selection.KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit"


# ═══ Reduce[C, A, **R = []]: Dataset → a ══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Evaluate(Reduce[Any, Tensor, [Callable[..., Tensor]]]):
    """A per-batch metric (accuracy, a loss, perplexity), averaged over every batch, weighted by their sizes; given a
    model, the metric is called as metric(model, *batch), as a loss is."""
    title = "Evaluation (a metric over a dataset)"

    def __call__(self, data: Dataset[Any], metric: Callable[..., Tensor], model: nn.Module | None = None, /) -> Tensor: ...


@dataclass(frozen=True)
class Metric(Reduce[tuple[Tensor, Tensor], Any]):
    """A metric of every prediction and target at once, not a mean of per-batch values: data is (prediction, target)
    pairs, gathered, then fn(targets, predictions), as scikit-learn's metrics take them."""
    title = "Whole-dataset metric"
    fn: Callable[[Any, Any], Any]

    def __call__(self, data: Dataset[tuple[Tensor, Tensor]], /) -> Any: ...


@dataclass(frozen=True)
class ClassificationMetrics(Metric):
    """The ones that need every score ranked or counted together: ROC AUC, F1, precision and recall, a confusion matrix."""
    title = "Classification metrics (AUC, F1, precision / recall, confusion matrix)"
    function = "sklearn.metrics.roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix"


@dataclass(frozen=True)
class Streaming(Reduce[tuple[Tensor, Tensor], Tensor]):
    """The same number as Metric without keeping every prediction: the metric is reset, updated on each (prediction,
    target) batch, then computed (torchmetrics' API)."""
    title = "Streaming metric"
    metric: StreamMetric

    def __call__(self, data: Dataset[tuple[Tensor, Tensor]], /) -> Tensor: ...


@dataclass(frozen=True)
class StreamingMetrics(Streaming):
    """Metrics of a whole dataset that keep running totals instead of every prediction: calibration error, FID."""
    title = "Calibration error, FID (streaming)"
    function = "torchmetrics.classification.CalibrationError, torchmetrics.image.fid.FrechetInceptionDistance"


@dataclass(frozen=True)
class FitPredict(Reduce[Any, Tensor]):
    """A fresh clone of the estimator fitted to the whole dataset, and what it says of each example: for methods that
    cannot label new data, only the data they were fitted to (fit_predict)."""
    title = "Fit and predict (transductive)"
    estimator: Estimator

    def __call__(self, data: Dataset[Any], /) -> Tensor: ...


@dataclass(frozen=True)
class DBSCAN(FitPredict):
    """Clusters as dense regions, how many found rather than given, some examples left as noise."""
    title = "Density clustering (DBSCAN, HDBSCAN)"
    function = "sklearn.cluster.DBSCAN, HDBSCAN, OPTICS"


@dataclass(frozen=True)
class HierarchicalClustering(FitPredict):
    """Clusters merged bottom up, the closest pair first, or cut from a graph of similarities (spectral)."""
    title = "Hierarchical and spectral clustering"
    function = "sklearn.cluster.AgglomerativeClustering, SpectralClustering"


@dataclass(frozen=True)
class FitTransform(Reduce[Any, Tensor]):
    """A fresh clone of the estimator fitted to the whole dataset, and each example's new features: for embeddings that
    place only the data they were fitted to (fit_transform)."""
    title = "Fit and transform (transductive)"
    estimator: Estimator

    def __call__(self, data: Dataset[Any], /) -> Tensor: ...


@dataclass(frozen=True)
class Embedding2D(FitTransform):
    """A low-dimensional map that keeps each example's neighbours near it: for looking at data, not for features."""
    title = "Neighbour embeddings (t-SNE, UMAP)"
    function = "sklearn.manifold.TSNE, umap.UMAP"


@dataclass(frozen=True)
class Accumulate(Reduce[Any, Any, [Scan[Any, Any, Any]]]):
    """A Scan's state carried across a dataset's batches, its final state returned: normalization statistics, a
    histogram, any metric that is not a mean."""
    title = "Accumulate (a scan across batches)"

    def __call__(self, data: Dataset[Any], s: Scan[Any, Any, Any], /) -> Any: ...


@dataclass(frozen=True)
class Peek(Reduce[Any, Any]):
    """The first batch of a dataset: what an expression can see of it, its type, to build a network or an index for."""
    title = "First batch"

    def __call__[C](self, data: Dataset[C], /) -> C: ...


# ═══ Phenomena: observed, not computed ════════════════════════════════════════════════════════════

phenomenon("Implicit regularization of SGD", "a property of training dynamics, seen in eval curves")
phenomenon("Double descent / benign overfitting", "seen by sweeping model or data size")
phenomenon("Grokking", "delayed generalization in long runs")
phenomenon("Superposition", "more features than dimensions; measured with sparse autoencoders")
phenomenon("General methods + scale beat hand-engineering", "an empirical lesson (the bitter lesson), not an op")

# ═══ Meta-primitives: each is a property of some groups ═══════════════════════════════════════════

meta("Bake in symmetry by sharing weights", (Aggregate,), "the same weights wherever the container relates elements: conv on a grid, attention on a set, messages on a graph")
meta("Let the input decide the computation", (Aggregate, Map), "attention weights, scan coefficients and expert choices computed from the data")
meta("Predict the hidden part from the visible part", (Split, Fill), "every self-supervised signal is a split into Holes; generation fills them back")
meta("Turn one hard step into many easy ones", (Fill, Walk, Unroll), "a token, a denoising step, an update at a time: a state, stepped")
meta("Learning is compression", (Map, Split), "bottlenecks, codebooks and students: the target, from a smaller description")
meta("Bootstrap from yourself", (Split,), "targets made by your own (older) model: TD, self-distillation, self-play, AlphaZero")
meta("Amortize, or search", (Map, Walk, Reshape, Search), "one forward pass that answers at once, or a tree grown by search")
meta("Every loss is a likelihood", (Score,), "Loss()(pred, target): what the target holds picks which")
meta("Not every objective is a loss", (Fold, Score), "a return is a fold of a path; its gradient comes from scoring the path, or there is none")
meta("Structure decides the computation", (Aggregate,), "the container picks what each element reads")

# ═══ Printing ═════════════════════════════════════════════════════════════════════════════════════

NEW = {"Tokenization (BPE, SentencePiece, bytes)", "Image tokenizer (VQ-VAE, VQGAN)", "Audio codec (EnCodec, residual quantization)",
       "Index (an empty one)", "Index training (IVF centroids, PQ codebooks)", "Index insertion", "Index deletion", "Codec fitting (BPE merges, a vocabulary)",
       "Collector (walks of a live policy)", "Replay buffer (empty)", "An expression on each batch", "Data filtering by fraction",
       "Evaluation (a metric over a dataset)", "Accumulate (a scan across batches)", "First batch", "Retrieval-augmented generation (retrieve)",
       "kNN-LM (a memory of contexts)", "Near-duplicate removal (MinHash / LSH)", "Hard-negative mining", "Fill", "Autoregressive order",
       "Unmasking (MaskGIT, masked diffusion)", "Sampling a flow (diffusion)", "Diffusion forcing", "Diffusion forcing sampling", "Training",
       "Optimizers (SGD, momentum, Adam, Muon)", "KV caching", "A path from a tree", "Sampling", "Several paths per root (best-of-N, groups)",
       "Teacher forcing (walking the data's path)", "Supervised", "Log-probability", "Retrieval accuracy", "Blank (where generation starts)",
       "A condition, fixed", "Objective (a split, then its loss)", "Minibatches", "Classifier-free guidance training", "Recurrence (RNN / LSTM / GRU)",
       "State-space models / linear recurrences", "Walk", "Classification", "Regression", "Linear models (linear and logistic regression)",
       "Kernel methods (random features)", "k-nearest neighbours (classification, regression)", "Hinge loss (SVM)",
       "Least squares (closed form, ridge)",
       # scikit-learn's API: estimators, their fitting, transductive methods, metrics
       "scikit-learn estimator", "Decision trees, random forests, gradient boosting", "Support vector machines (kernel SVM)",
       "Naive Bayes", "Gaussian processes", "k-means", "Mixture models (GMM, EM)", "PCA (dimensionality reduction)",
       "Standardization (feature scaling)", "Fitting an estimator (fit, partial_fit)", "Fit and predict (transductive)",
       "Density clustering (DBSCAN, HDBSCAN)", "Hierarchical and spectral clustering", "Fit and transform (transductive)",
       "Neighbour embeddings (t-SNE, UMAP)", "Whole-dataset metric", "Classification metrics (AUC, F1, precision / recall, confusion matrix)",
       "Streaming metric", "Calibration error, FID (streaming)",
       # sub-datasets: sampling and splitting
       "Random subsampling", "Resampling (importance sampling, bootstrap)", "Class rebalancing (over- and undersampling)",
       "Train / validation / test split", "Splits from a splitter", "k-fold cross-validation"}


def _paint(code: str) -> Callable[[str], str]:
    """An ANSI colour, or nothing when the output isn't a terminal (or NO_COLOR is set)."""
    import os
    import sys

    on = (sys.stdout.isatty() or os.environ.get("FORCE_COLOR")) and not os.environ.get("NO_COLOR")
    return (lambda t: f"\033[{code}m{t}\033[0m") if on else (lambda t: t)


def _about(g: type[Method]) -> str:
    return g.__dict__.get("about") or " ".join((inspect.getdoc(g) or "").split("\n\n")[0].split())


def _value(v: Any) -> str:
    """A default as written: "bpe", 0.1, Optimizer()."""
    return f'"{v}"' if isinstance(v, str) else repr(v)


def _parameters_text(fn: Callable[..., Any]) -> str:
    """A method's parameters as written, without self: (paths: Path, scores: Value, /). A group's own *rest: R.args,
    **named: R.kwargs is written *R: whatever else its methods take."""
    out, slash = [], False
    params = list(inspect.signature(fn).parameters.values())[1:]
    for p in params:
        if p.kind == p.VAR_KEYWORD and str(p.annotation).endswith(".kwargs"):
            continue
        if slash and p.kind != p.POSITIONAL_ONLY:
            out.append("/")
            slash = False
        slash = slash or p.kind == p.POSITIONAL_ONLY
        if p.kind == p.VAR_POSITIONAL and str(p.annotation).endswith(".args"):
            out.append("*" + str(p.annotation).removesuffix(".args"))
            continue
        star = "*" if p.kind == p.VAR_POSITIONAL else "**" if p.kind == p.VAR_KEYWORD else ""
        text = f"{star}{p.name}" + (f": {p.annotation}" if p.annotation is not inspect.Parameter.empty else "")
        out.append(text + (f" = {_value(p.default)}" if p.default is not inspect.Parameter.empty else ""))
    if slash:
        out.append("/")
    return f"({', '.join(out)})"


def _methods(cls: type) -> list[tuple[str, Callable[..., Any]]]:
    """The methods a class defines itself, its __call__ first: what it does to data, and a scan's init and step."""
    own = [(n, f) for n, f in cls.__dict__.items() if inspect.isfunction(f) and (n == "__call__" or not n.startswith("_"))]
    return sorted(own, key=lambda nf: nf[0] != "__call__")


def _signature_text(name: str, fn: Callable[..., Any]) -> str:
    """name(params) -> returns, for __call__ just (params) -> returns, its type parameters shown when it has some."""
    tparams = getattr(fn, "__type_params__", ())
    generic = f"[{', '.join(_type_parameter(t) for t in tparams)}]" if tparams else ""
    ret = inspect.signature(fn).return_annotation
    return f"{'' if name == '__call__' else name}{generic}{_parameters_text(fn)} -> {ret}"


def _type_parameter(t: Any) -> str:
    """C, **R = [], In = C, C: Container: a type parameter as written in a class header."""
    star = "**" if isinstance(t, ParamSpec) else "*" if isinstance(t, TypeVarTuple) else ""
    bound = getattr(t, "__bound__", None)
    text = f"{star}{t.__name__}" + (f": {getattr(bound, '__name__', bound)}" if bound is not None else "")
    default = getattr(t, "__default__", NoDefault)
    if default is not NoDefault:
        shown = "[" + ", ".join(map(_type_text, default)) + "]" if isinstance(default, (list, tuple)) else _type_text(default)
        text += f" = {shown}"
    return text


def _header(cls: type) -> str:
    """A class header as written in this file: BestOfN(Fold[Path, Path, [Value]]), Fold[C, A, **R = []]."""
    if "group" in cls.__dict__:
        params = getattr(cls, "__type_params__", ())
        return cls.__name__ + (f"[{', '.join(map(_type_parameter, params))}]" if params else "")
    source = inspect.getsource(cls)
    return source[source.index("class ") + 6 : source.index(":\n")]


def _settings_text(cls: type) -> str:
    """Its settings, as written: n: int, kind: str = "bpe"."""
    out = []
    for f in fields(cls) if is_dataclass(cls) else ():
        default = (_value(f.default) if f.default is not MISSING else
                   _value(f.default_factory()) if f.default_factory is not MISSING else None)
        out.append(f"{f.name}: {f.type}" + (f" = {default}" if default is not None else ""))
    return ", ".join(out)


def _members(base: type[Method]) -> list[Entry]:
    """The methods whose group is base: base itself (Fill, Walk, Unroll are their own group), then its subclasses."""
    return [e for e in ENTRIES if e.cls is not None and group_class(e.cls) is base]


def print_catalog(explain: bool = False, short: bool = False) -> None:
    """By framework, family and group. Each group's base classes, with the type parameters and the __call__ every
    method of theirs agrees with (pyright checks it); under each, its methods: the class as declared, its title, its
    own __call__ (a scan's init and step too) and its settings. explain adds what each does; short is one line each."""
    bold, cyan, yellow, magenta, dim, green = _paint("1"), _paint("1;36"), _paint("33"), _paint("35"), _paint("2"), _paint("32")
    for heading, families in FRAMEWORKS.values():
        print(bold(f"\n━━━━ {heading} ━━━━"))
        for family in families:
            print(bold(f"\n══ {FAMILIES[family]} ══"))
            for bases in (b for b in GROUPS.values() if b[0].family == family):
                for base in bases:
                    print(f"\n{cyan(_header(base))}  {dim(base.form)}")
                    if "__call__" in base.__dict__ and "title" not in base.__dict__:
                        for name, fn in _methods(base):
                            print(f"  {yellow(_signature_text(name, fn))}")
                    print(dim(textwrap.fill(_about(base), 110, initial_indent="  ", subsequent_indent="  ")))
                    for e in _members(base):
                        cls = e.cls
                        assert cls is not None
                        mark = green("★ ") if e.name in NEW else ""
                        if short:
                            state = f"  {dim('·')} {magenta('state: ' + e.state)}" if e.state else ""
                            print(f"  - {mark}{bold(e.name)}: {yellow(e.signature)}{state}")
                            continue
                        name = cls.__name__ if cls is base else _header(cls).replace(cls.__name__, bold(cls.__name__), 1)
                        print(f"\n  {name}  {mark}{dim(e.name)}")
                        if cls.function:
                            print(f"    {dim('→ ' + cls.function)}")
                        for mname, fn in _methods(cls):
                            print(f"    {yellow(_signature_text(mname, fn))}")
                        if is_dataclass(cls) and fields(cls):
                            print(magenta(textwrap.fill(_settings_text(cls), 110, initial_indent="    ", subsequent_indent="      ")))
                        if cls.state:
                            print(f"    {dim('state: ' + cls.state)}")
                        if explain:
                            print(dim(textwrap.fill(e.explanation, 110, initial_indent="    ", subsequent_indent="    ")))
    print(bold("\n══ META-PRIMITIVES · properties of the groups ══\n"))
    for e in (e for e in ENTRIES if e.group == "meta"):
        print(f"  - {bold(e.name)}: {cyan(e.signature)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    print(bold("\n══ PHENOMENA · observed, not computed ══\n"))
    for e in (e for e in ENTRIES if e.group == "phenomenon"):
        print(f"  - {bold(e.name)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    rows = [e for e in ENTRIES if e.group in GROUPS]
    stmts = [e for e in rows if e.cls is not None and issubclass(e.cls, Stmt)]
    bases = [b for bs in GROUPS.values() for b in bs]
    print(dim(f"\n{len(rows)} methods in {len(GROUPS)} groups ({len(bases)} base classes), {sum(bool(e.state) for e in rows)} carrying a state · "
              f"{len(stmts)} statements ({sum(e.changes is not None for e in stmts)} changing their argument) · "
              f"{sum(e.side == 'agnostic' for e in rows)} model-agnostic, {sum(e.side == 'model' for e in rows)} model-dependent"))


if __name__ == "__main__":
    import sys

    print_catalog(explain="--explain" in sys.argv, short="--short" in sys.argv)
