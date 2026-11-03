"""
catalog.py: every method in the original list, as a class. A method is a frozen dataclass: its fields are its
settings, its __call__ takes the data, its docstring says what it does, and the group it subclasses says what kind
of thing it is, with the group's type parameters filled in:

    class Best(Fold[Path, Path]):           # best-of-N is a fold: Paths in, a Path out
        n: int = 8                          # a setting
        def __call__(self, paths: Path, scores: Value, /) -> Path: ...

    Best(n=16)(paths, scores)               # configure, then call

Expressions take batches (a Container, in memory), build values and change nothing: queries. Statements take
datasets, or change something in place: writes, and the blocking work that needs every batch. The interface only:
bodies are left to the implementation (torch, torchrl, your own).

EXPRESSIONS · on a batch, nothing changes
  structure   Encode     A ⇄ C[classes]        a codec: raw data in a vocabulary and back
              Convert    C[a] → C′[a]          change the container, keep what the elements are
              Map        C[a] → C[b]           change each element on its own, keep the container
              Aggregate  C[h] → C[h]           each element reads the elements it is related to
              Select     C[a] → C[a]           keep, drop, reorder or reweight elements
              Fold       C[a] → a              reduce a container to one value
              Scan       Seq[X] → Seq[Y]       carry a state along a sequence: init, then step
  learning    Split      C → Example           hide part of a container (Holes); what was hidden is the target
              Score      Prediction × C → ℝ    how bad a prediction is: every loss, every surrogate objective
  partial     Start      … → Partial           a partial container to grow: a tree, holes
              Fill       Holes[C] → C          fill the holes, an Order deciding which: a split, undone
              Order      Holes → Holes         which holes a fill fills next, and how far
              Walk       PartialTree → Path    grow a tree's known part, a Choose deciding which leaves to keep
              Choose     PartialTree → …       a walk's step: which edges to follow, which leaves to keep
              Unroll     PartialSeq[S] → S     step a state until done
              Evolve     S → S                 an unroll's step, where there is no gradient to follow
              Reshape    Partial → Partial     change how a partial container goes on: its options, its model
  lookup      Search     Index × Value → Set   what is built on index.search: retrieval, kNN, negatives
  models      Wrap       Module → Module′      a new model around an old one, which stays as it was

STATEMENTS · on a dataset, or changing something in place (!)
  update      Fit        X! × Dataset          fit something to a dataset: a model by a loss, a codec by its rule
              Assign     Weights!              set parameter values directly: copied in, initialized, averaged
              Alter      Module!               change what the parameters are: add, freeze, quantize, shard
              Store      Index! / Buffer!      keep a store up to date: train an index, insert, delete, extend
  dataset     Source     … → Dataset           where batches come from: memory, an environment, a buffer
              Curate     Dataset → Dataset     keep, drop or reorder examples looking at all of them (blocking)
              Reduce     Dataset → a           one value from a whole dataset: a metric, a statistic, a batch

A statement is never a value inside an expression: in a script (program.py) it is a line of its own, as a SQL
write or CREATE INDEX sits at the top of a statement. Update statements change the argument marked ! and return
None; dataset statements change nothing but read every batch. Weights is the subtype of Module whose architecture an
op leaves alone.

The groups are classes, and a group's type parameters say what each method works on: Best is a Fold[Path, Path],
Network a Map[Container, Prediction] (it builds a model from containers to predictions), Train a Fit[nn.Module]
(what it changes). Registering a method checks its __call__ against them. Where every method of a group shares one
signature (Choose, Order, Evolve, Scan), the group declares it and the methods inherit it; Fill, Walk and Unroll
are each one method, and the group is that method.

What a statement changes is never a config: a method that makes a model, a codec, an index or a buffer is a config
whose __call__ returns a new one (Network(d=64)(x, y), Tokenizer(vocab=32_000)()). Everything a config holds is a
setting.

Each method is model-agnostic (any model is an opaque function to it, or it uses none) or model-dependent (it is
a layer, or it opens a model up). Phenomena are observed rather than computed; the meta-primitives are
properties of the groups; the well-known works are programs of methods (program.py).

    python catalog.py             prints the catalog, one line per method
    python catalog.py --explain   with what each one does
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, dataclass_transform, get_args, get_origin

import torch
from torch import Tensor, nn

from data import (
    Buffer, Chooser, Codec, Container, Dataset, Dist, Environment, Example, Graph, Grid, Holes, Index, Model, PartialSeq, PartialTree, Path,
    Prediction, Predictor, Product, Schedule, Seq, Set, Value, Weights,
)

# ═══ Methods and groups ═══════════════════════════════════════════════════════════════════════════


@dataclass_transform(kw_only_default=True, frozen_default=True)
class Method:
    """The root of the catalog. Subclassing a group makes a method: a frozen dataclass (keyword-only fields, its
    settings) whose __call__ takes the data. A class with a title is a method; one with a group is a group."""

    title: ClassVar[str] = ""             # how the catalog prints a method
    state: ClassVar[str] = ""             # what it keeps, when its type carries a state
    side: ClassVar[str] = ""              # "agnostic" or "model"; the group's when empty
    changes: ClassVar[int | None] = None  # an update: which argument of __call__ it changes in place
    function: ClassVar[str] = ""          # a method torch already has: its name there (nothing to call here)
    signature: ClassVar[str] = ""         # such a method's signature, when not its group's form

    __dataclass_fields__: ClassVar[dict[str, Any]]   # every subclass is made a dataclass, below

    group: ClassVar[str] = ""             # a group: its name, its family, its form; about says what it does
    family: ClassVar[str] = ""
    form: ClassVar[str] = ""
    about: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True, kw_only=True, repr=False)(cls)
        if "group" in cls.__dict__:
            _register_group(cls)
        if "title" in cls.__dict__:
            _register_method(cls)

    def __repr__(self) -> str:
        """The class and the settings that differ from their defaults: Beam(k=4), Greedy()."""
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in settings_of(self).items())})"


class Expr(Method):
    """An expression: on batches in memory; it changes nothing."""


class Stmt(Method):
    """A statement: on a dataset, or changing its argument (!) in place and returning None. A line of its own in a
    script, never inside an expression."""


def settings_of(config: Method) -> dict[str, Any]:
    """A config's settings that differ from their defaults (all of the required ones)."""
    return {f.name: getattr(config, f.name) for f in fields(config) if _differs(getattr(config, f.name), f.default)}


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
            "update": "UPDATE · a model, a codec, an index or a buffer, changed in place",
            "dataset": "DATASET · read every batch, change nothing"}
FRAMEWORKS = {"expr": ("EXPRESSIONS · on a batch, nothing changes", ["structure", "learning", "partial", "lookup", "models"]),
              "stmt": ("STATEMENTS · on a dataset, or changing something in place (!)", ["update", "dataset"])}

GROUPS: dict[str, type[Method]] = {}     # name → its class, in the order they are defined


@dataclass(frozen=True)
class Entry:
    group: str                  # a key of GROUPS, or "phenomenon" / "meta"
    name: str                   # its title
    signature: str              # its type; for "meta", the groups it is about
    state: str                  # what it carries, and what it keeps there ("" when it carries none)
    side: str                   # "agnostic" | "model" | "" (phenomena, meta)
    settings: tuple[str, ...]   # its fields
    function: str               # its class here, or torch's name for it
    explanation: str
    cls: type[Method] | None = None
    changes: int | None = None  # an update: which argument of __call__ it changes in place


ENTRIES: list[Entry] = []
_BY_CLASS: dict[type, Entry] = {}


def _register_group(cls: type[Method]) -> None:
    if cls.family not in FAMILIES:
        raise TypeError(f"{cls.__name__}: unknown family {cls.family!r}")
    GROUPS[cls.group] = cls


def group_class(cls: type) -> type[Method]:
    """The group a method belongs to: the nearest class in its ancestry that declares one."""
    return next(k for k in cls.__mro__ if "group" in k.__dict__)


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
        _check_parameters(cls, g, ins, out)
        signature = " × ".join(ins) or "()" if cls.changes is not None else f"{' × '.join(ins) or '()'} → {out}"
        state = f"{state_type} — {cls.state}" if state_type and cls.state else state_type or cls.state
    doc = inspect.getdoc(cls)
    if not doc or doc == inspect.getdoc(g) and cls is not g:
        raise TypeError(f"{cls.__name__}: write what it does in its docstring")
    entry = Entry(g.group, cls.title, signature, state, cls.side or g.side, tuple(f.name for f in fields(cls)),
                  cls.function or cls.__name__, " ".join(doc.split("\n\n")[0].split()), cls, cls.changes)
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
    call = cls.__call__
    bounds = {t.__name__: getattr(t.__bound__, "__name__", t.__name__) for t in getattr(call, "__type_params__", ()) if t.__bound__ is not None}
    owner = next(k for k in cls.__mro__ if "__call__" in k.__dict__)   # an inherited call reads the class's parameters
    bounds |= {t.__name__: _type_text(a) for t, a in zip(getattr(owner, "__type_params__", ()), _parameters(cls, owner))}

    def short(annotation: Any) -> str:
        return _short(re.sub(r"\b\w+\b", lambda m: bounds.get(m.group(), m.group()), str(annotation)))

    sig = inspect.signature(call)
    params = [p for p in list(sig.parameters.values())[1:]
              if p.kind == p.VAR_POSITIONAL or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect.Parameter.empty)]
    ins = [short(p.annotation) + ("…" if p.kind == p.VAR_POSITIONAL else "") + ("!" if i == cls.changes else "") for i, p in enumerate(params)]
    return ins, short(sig.return_annotation), ""


def _parameters(cls: type, base: type) -> tuple[Any, ...]:
    """The type parameters cls gives base: (Path, Path) for Best(Fold[Path, Path]); () when it gives none."""
    for k in cls.__mro__:
        for b in k.__dict__.get("__orig_bases__", ()):
            if get_origin(b) is base:
                return get_args(b)
    return ()


def _check_parameters(cls: type[Method], g: type[Method], ins: list[str], out: str) -> None:
    """A method's group parameters agree with its __call__. [In, Out]: Out is what it returns (or, for one that builds a
    model or a codec, what that maps to), In one of its inputs (or what the model maps from). [X] of an update: what it
    changes; [D] of a source: what it returns; [C] of a curate: what its datasets hold; [P] of a start: what it returns."""
    given = [_short(_type_text(a)) for a in _parameters(cls, g)]
    if not given or "__call__" in g.__dict__:       # the group declares the call: the type checker checks the methods
        return
    plain = [i.rstrip("!…") for i in ins]
    built = re.fullmatch(r"(?:Model|Codec)\[(.*) [→⇄] (.*)\]", out)
    outs = {out, "Any"} | ({built.group(2)} if built else set()) | ({"Prediction"} if out == "Predictor" else set())
    froms = set(plain) | ({built.group(1)} if built else set()) | {"Any"}

    def same(a: str, b: str) -> bool:   # equal, or either is Any, or one is the other without its parameters
        return a == b or "Any" in (a, b) or (a.split("[")[0] == b.split("[")[0] and ("[" not in a or "[" not in b))

    def among(a: str, options: set[str]) -> bool:
        return any(same(a, o) for o in options)

    if g in (Fit, Assign, Alter, Store):
        ok = cls.changes is not None and same(plain[cls.changes], given[0])
    elif g in (Source, Start):
        ok = same(given[0], out)
    elif g is Curate:
        ok = same(out, f"Dataset[{given[0]}]")
    elif g is Split:
        ok = among(given[0], froms)
    else:
        ok = among(given[1], outs) and among(given[0], froms)
    if not ok:
        raise TypeError(f"{cls.__name__}({g.__name__}[{', '.join(given)}]) does not match its __call__: {' × '.join(ins) or '()'} → {out}")


def entry_of(f: Any) -> Entry | None:
    """The entry of a method: its class, or a config of it."""
    return _BY_CLASS.get(f if isinstance(f, type) else type(f))


def changes(f: Any) -> int | None:
    """Which argument of its __call__ f changes in place, when f is an update (None otherwise)."""
    e = entry_of(f)
    return e.changes if e is not None else None


def is_statement(f: Any) -> bool:
    """Whether f is a statement: it reads a dataset, or changes something; either way, never inside an expression."""
    cls = f if isinstance(f, type) else type(f)
    return isinstance(cls, type) and issubclass(cls, Stmt) and cls in _BY_CLASS


def phenomenon(name: str, explanation: str) -> None:
    ENTRIES.append(Entry("phenomenon", name, "", "", "", (), "", explanation))


def meta(name: str, groups: tuple[type[Method], ...], explanation: str) -> None:
    ENTRIES.append(Entry("meta", name, ", ".join(g.group for g in groups), "", "", (), "", explanation))


# ═══ Expression groups ════════════════════════════════════════════════════════════════════════════


class Encode[A, B](Expr):
    """A codec: write raw data A in a vocabulary B and read it back. A method builds one; FitCodec fits one that learns."""
    group, family, form, side = "encode", "structure", "A ⇄ C[classes]", "model"


class Convert[A, B](Expr):
    """Change the container, keep what the elements are."""
    group, family, form, side = "convert", "structure", "C[a] → C′[a]", "agnostic"


class Map[A, B](Expr):
    """Change each element on its own, keep the container."""
    group, family, form, side = "map", "structure", "C[a] → C[b]", "model"


class Aggregate[A, B](Expr):
    """Each element reads the elements it is related to."""
    group, family, form, side = "aggregate", "structure", "C[h] → C[h]", "model"


class Select[A, B](Expr):
    """Keep, drop, reorder or reweight elements."""
    group, family, form, side = "select", "structure", "C[a] → C[a]", "agnostic"


class Fold[A, B](Expr):
    """Reduce a container to one value."""
    group, family, form, side = "fold", "structure", "C[a] → a", "agnostic"


class Scan[X, S, Y](Expr):
    """Carry a state along a sequence: init gives the first state, step takes a state and an element to the next
    state and an output; calling it runs the steps along a sequence (from the end when reverse)."""
    group, family, form, side = "scan", "structure", "Seq[X] → Seq[Y]", "agnostic"
    reverse: ClassVar[bool] = False

    def init(self) -> S: ...

    def step(self, state: S, x: X, /) -> tuple[S, Y]: ...

    def __call__(self, xs: Seq, /) -> Seq: ...


class Split[A](Expr):
    """Hide part of a container; what was hidden (or a teacher's answer) is the target. A is what it splits."""
    group, family, form, side = "split", "learning", "C → Example", "agnostic"


class Score[A, B](Expr):
    """How bad a prediction is: every loss, every surrogate objective."""
    group, family, form, side = "score", "learning", "Prediction × C → ℝ", "agnostic"


class Start[P](Expr):
    """A partial container to grow, P: a tree for a walk, holes for a fill."""
    group, family, form, side = "start", "partial", "… → Partial", "agnostic"


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


class Reshape[A, B](Expr):
    """Change how a partial container goes on: its options, its model."""
    group, family, form, side = "reshape", "partial", "Partial → Partial", "agnostic"


class Search[A, B](Expr):
    """What is built on looking up an index (index.search): the k elements nearest each query, put to use."""
    group, family, form, side = "search", "lookup", "Index × Value → Set", "agnostic"


class Wrap[A, B](Expr):
    """A new model around an old one, which stays as it was."""
    group, family, form, side = "wrap", "models", "Module → Module′", "agnostic"


# ═══ Statement groups ═════════════════════════════════════════════════════════════════════════════


class Fit[X](Stmt):
    """Fit something to a dataset, in place: a model by a loss's gradient, a codec by its own rule. X is what it changes."""
    group, family, form, side = "fit", "update", "X! × Dataset", "agnostic"
    changes = 0


class Assign[X](Stmt):
    """Set parameter values directly, without a gradient: copied in, initialized, averaged."""
    group, family, form, side = "assign", "update", "Weights!", "agnostic"
    changes = 0


class Alter[X](Stmt):
    """Change what the parameters are: add some, freeze some, quantize, shard."""
    group, family, form, side = "alter", "update", "Module!", "model"
    changes = 0


class Store[X](Stmt):
    """Keep a store up to date: learn an index's structure, insert into it, delete from it, add to a buffer."""
    group, family, form, side = "store", "update", "Index! / Buffer!", "agnostic"
    changes = 0


class Source[D](Stmt):
    """Where batches come from: memory, an environment, a buffer, an expression on each batch. D is the dataset."""
    group, family, form, side = "source", "dataset", "… → Dataset", "agnostic"


class Curate[C](Stmt):
    """Keep, drop or reorder examples, looking at all of them (blocking). C is the batch type, kept."""
    group, family, form, side = "curate", "dataset", "Dataset → Dataset", "agnostic"


class Reduce[A, B](Stmt):
    """One value from a whole dataset: a metric over every batch, a statistic, one batch."""
    group, family, form, side = "reduce", "dataset", "Dataset → a", "agnostic"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# EXPRESSIONS · on a batch, nothing changes
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ═══ Encode[A, B]: A ⇄ C[classes] ═════════════════════════════════════════════════════════════════


class Tokenizer(Encode[list[str], Seq]):
    """Text ⇄ a sequence of ids, built empty: BPE merges, fitted to a corpus by FitCodec (bytes alone: a fixed
    vocabulary of 256, nothing to fit). The tokenizer decides the content type: Seq of classes(vocab)."""
    title = "Tokenization (BPE, SentencePiece, bytes)"
    kind: str = "bpe"
    vocab: int = 32000

    def __call__(self) -> Codec[list[str], Seq]: ...


class ImageTokenizer(Encode[Grid, Grid]):
    """An image ⇄ a coarser grid of codebook ids: an encoder, vector quantization, a decoder, trained to reconstruct."""
    title = "Image tokenizer (VQ-VAE, VQGAN)"
    codes: int = 1024

    def __call__(self, images: Grid, /) -> Codec[Grid, Grid]: ...


class AudioCodec(Encode[Seq, Product]):
    """A waveform ⇄ frames × codebooks of ids: each codebook quantizes what the previous ones left (residual VQ)."""
    title = "Audio codec (EnCodec, residual quantization)"
    codebooks: int = 8
    codes: int = 1024

    def __call__(self, audio: Seq, /) -> Codec[Seq, Product]: ...


# ═══ Convert[A, B]: C[a] → C′[a] ══════════════════════════════════════════════════════════════════


class Patchify(Convert[Grid, Grid]):
    """An image → a coarser grid of p×p patches, each one element (a reshape); a linear map after it is a patch
    embedding."""
    title = "Patching"
    size: int

    def __call__(self, x: Grid, /) -> Grid: ...


class Flatten(Convert[Grid, Seq]):
    """A grid → a sequence in raster order: the lattice is forgotten, learned positions say where each element was."""
    title = "Flattening"

    def __call__(self, x: Grid, /) -> Seq: ...


class Concat(Convert[Seq, Seq]):
    """Several sequences of the same width as one (image patches, then caption tokens): one network reads them all."""
    title = "Multimodal sequence"

    def __call__(self, *parts: Seq) -> Seq: ...


class RandomFlip(Convert[Grid, Grid]):
    """Mirror each image with probability p: its elements move, none changes."""
    title = "Data augmentation: random flip"
    p: float = 0.5

    def __call__(self, x: Grid, /) -> Grid: ...


class RandomCrop(Convert[Grid, Grid]):
    """Pad, and cut back at a random offset: a random translation."""
    title = "Data augmentation: random crop"
    padding: int = 2

    def __call__(self, x: Grid, /) -> Grid: ...


class Transitions(Convert[Path, Value]):
    """The steps of walked paths as a set of transitions (state, action, reward, next state, done, and the action's
    log-probability), their order dropped: what a replay buffer holds and RL losses read."""
    title = "Experience (transitions)"

    def __call__(self, path: Path, returns: Seq | None = None, /) -> Value: ...


class PathTo(Convert[PartialTree[Any, Any], Path]):
    """The steps from the root to a leaf of a tree's known part: the result of a walk."""
    title = "A path from a tree"

    def __call__(self, tree: PartialTree[Any, Any], leaf: Tensor, /) -> Path: ...


class InContextLearning(Convert[Seq, Seq]):
    """Solved examples, then the query, as one prompt: the model adapts by reading them."""
    title = "In-context learning (few-shot prompts)"
    function, signature = "Concat", "Seq… → Seq"


# ═══ Map[A, B]: C[a] → C[b] ═══════════════════════════════════════════════════════════════════════


class Network(Map[Container, Prediction]):
    """embed → position → [aggregate, MLP] × depth → pool → readout, each slot chosen from the input and target types
    (EmbedFor, AggregateFor, PoolFor, ReadoutFor); mixer and mlp replace a slot's choice with a config (a scan,
    experts). Calling it builds the network for inputs like x and targets like y."""
    title = "Network (each layer chosen by the types)"
    d: int = 64
    depth: int = 2
    mixer: Aggregate[Any, Any] | None = None
    mlp: Map[Any, Any] | None = None

    def __call__(self, x: Container, y: Container, /) -> Predictor: ...


class Encoder(Map[Container, Value]):
    """A network from any container to one vector per example: the towers of CLIP, a reward model, a critic."""
    title = "Embedding models (a container to one vector)"
    dim: int

    def __call__(self, x: Container, /) -> Model[Container, Value]: ...


class EmbedFor(Map[Container, Container]):
    """By what the elements hold: classes → a row of a lookup table (one more for [MASK]); real values → a linear map."""
    title = "Embeddings"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


class ReadoutFor(Map[Container, Container]):
    """By what the target holds: a logit per class, a value per real number, one readout per field of a record."""
    title = "Readout"
    d: int

    def __call__(self, y: Container, /) -> Model[Container, Container]: ...


class LearnedPositions(Map[Container, Container]):
    """A learned vector added at each position of a Seq, a Grid or a Product; a Set, a Graph or a Tree gets none."""
    title = "Positional encoding: learned"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


class SinusoidalPositions(Map[Container, Container]):
    """Fixed sin/cos features of each position at geometrically spaced frequencies, added."""
    title = "Positional encoding: sinusoidal"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


class RoPE(Map[Tensor, Tensor]):
    """Rotate pairs of query and key features by angles proportional to the position, so that attention depends only
    on relative offsets."""
    title = "Positional encoding: rotary (RoPE)"

    def __call__(self, x: Tensor, positions: Tensor, /) -> Tensor: ...


class MoE(Map[Container, Container]):
    """A router sends each element to its top-k experts (MLPs) and sums their gate-weighted outputs."""
    title = "Mixture of experts"
    d: int
    experts: int = 8
    k: int = 2

    def __call__(self) -> Model[Container, Container]: ...


class VectorQuantize(Map[Container, Container]):
    """Each element → the id of its nearest codebook entry (real → classes), with a straight-through gradient."""
    title = "Vector quantization"

    def __call__(self, x: Container, codebook: Tensor, /) -> Container: ...


class StraightThrough(Map[Tensor, Tensor]):
    """Forward the hard value, backpropagate through the soft one: a gradient for a choice that has none."""
    title = "Straight-through estimator"
    side = "agnostic"

    def __call__(self, hard: Tensor, soft: Tensor, /) -> Tensor: ...


class Autoencoder(Map[Container, Container]):
    """encode to a smaller container (fewer positions, few channels), decode back: the latent must keep what matters,
    and generation can happen there (latent diffusion). Its .encode and .decode are the two halves."""
    title = "Latent space / bottleneck"
    latent: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


class Condition(Map[Container, Container]):
    """Attach a label to every element (its one-hot vector next to each element's values; a none label for guidance)."""
    title = "Conditioning"
    side = "agnostic"

    def __call__(self, x: Container, label: Value, /) -> Container: ...


class SparseAutoencoder(Map[Container, Container]):
    """Activations → a wide, sparse code → activations, trained by reconstruction plus an L1 penalty: each code unit
    tends to be one feature."""
    title = "Sparse autoencoders (superposition)"
    d: int
    features: int

    def __call__(self) -> Model[Container, Container]: ...


class LinearProbe(Map[Value, Value]):
    """A linear readout fitted on frozen activations: is a feature readable along one direction?"""
    title = "Linear probes (features as directions)"

    def __call__(self, acts: Value, labels: Value, /) -> Model[Value, Value]: ...


class ColorJitter(Map[Grid, Grid]):
    """Scale each image's brightness and contrast by random factors."""
    title = "Data augmentation: color jitter"
    side = "agnostic"
    strength: float = 0.2

    def __call__(self, x: Grid, /) -> Grid: ...


class GaussianNoise(Map[Container, Container]):
    """x + σ·ε at every element."""
    title = "Data augmentation: additive noise"
    side = "agnostic"
    sigma: float = 0.05

    def __call__(self, x: Container, /) -> Container: ...


class LabelSmoothing(Map[Value, Value]):
    """A class → a distribution over classes, (1 − ε)·one_hot + ε / n: never pushed to infinite confidence."""
    title = "Label smoothing"
    side = "agnostic"
    eps: float = 0.1

    def __call__(self, y: Value, /) -> Value: ...


class Attribution(Map[Predictor, Container]):
    """How much each input element pushed the output: the input times the output's gradient, on any differentiable
    model."""
    title = "Attribution (input × gradient)"
    side = "agnostic"

    def __call__(self, model: Predictor, x: Container, /) -> Container: ...


class Chinchilla(Map[float, tuple[float, float]]):
    """A compute budget → (parameters, tokens): C ≈ 6·N·D with D ≈ 20 tokens per parameter."""
    title = "Compute-optimal allocation (Chinchilla)"
    side = "agnostic"

    def __call__(self, compute: float, /) -> tuple[float, float]: ...


class MLP(Map[Container, Container]):
    """The same small network on every element."""
    title = "Linear layer + nonlinearity (MLP)"
    function, signature = "nn.Linear, nn.GELU, nn.Sequential", "C[h] → C[h]"


class SwiGLU(Map[Container, Container]):
    """silu(gate(h)) · up(h), then down: one branch decides how much of the other passes."""
    title = "Gating (SwiGLU)"
    function, signature = "nn.SiLU", "C[h] → C[h]"


class Normalization(Map[Container, Container]):
    """Rescale each element's features (BatchNorm aggregates over the batch instead)."""
    title = "Normalization"
    function, signature = "nn.RMSNorm, nn.LayerNorm", "C[h] → C[h]"


class Dropout(Map[Container, Container]):
    """Zero random features while training."""
    title = "Dropout"
    function, signature = "nn.Dropout", "C[h] → C[h]"


# ═══ Aggregate[A, B]: C[h] → C[h] ═════════════════════════════════════════════════════════════════


class AggregateFor(Aggregate[Container, Container]):
    """What each element reads, from the container: Graph, Tree → message passing · Product → axial · causal Seq →
    causal attention (a selective scan when long) · large Grid → convolution · else attention over all."""
    title = "Choosing the aggregation from the container"
    d: int

    def __call__(self, x: Container, /) -> Model[Container, Container]: ...


class Attention(Aggregate[Container, Container]):
    """Each element reads every other (or every earlier one), weighted by query·key similarity."""
    title = "Attention (self, causal)"
    d: int
    causal: bool = False

    def __call__(self) -> Model[Container, Container]: ...


class CrossAttention(Aggregate[tuple[Container, Container], Container]):
    """Queries from one container, keys and values from another (text reading an image)."""
    title = "Cross-attention"
    d: int

    def __call__(self) -> Model[tuple[Container, Container], Container]: ...


class Conv(Aggregate[Grid, Grid]):
    """Each element reads its neighbours on the lattice, with the same weights at every offset."""
    title = "Convolution"
    d: int
    size: int = 3

    def __call__(self) -> Model[Grid, Grid]: ...


class Recurrence(Aggregate[Seq, Seq]):
    """Each element reads a state carried from the earlier ones, updated by a gated nonlinear function."""
    title = "Recurrence (RNN / LSTM / GRU)"
    state = "a hidden vector, carried along the sequence"
    d: int

    def __call__(self) -> Model[Seq, Seq]: ...


class SelectiveScan(Aggregate[Seq, Seq]):
    """A linear recurrence whose coefficients come from the input (S4, Mamba, linear attention): every earlier element
    read, in linear time."""
    title = "State-space models / linear recurrences"
    state = "s_t = a_t·s_{t−1} + b_t, run as a parallel scan"
    d: int

    def __call__(self) -> Model[Seq, Seq]: ...


class MessagePassing(Aggregate[Graph, Graph]):
    """Each node reads its own features and the mean along its incoming edges; in a tree, children pass up to parents."""
    title = "Message passing (GNN)"
    d: int

    def __call__(self) -> Model[Graph, Graph]: ...


class Axial(Aggregate[Product, Product]):
    """One aggregation per factor of a product (frames, then pixels), the other factors as batch; summed."""
    title = "Axial (products of structures)"
    d: int
    parts: tuple[str, ...] = ('Seq', 'Grid')

    def __call__(self) -> Model[Product, Product]: ...


class UNet(Aggregate[Grid, Grid]):
    """Down to a coarse grid and back up, with skip connections: each element reads a wide neighbourhood."""
    title = "Encoder–decoder hierarchy (U-Net)"
    d: int
    levels: int = 2

    def __call__(self) -> Model[Grid, Grid]: ...


class Retrieval(Aggregate[tuple[Container, Index], Container]):
    """Each element reads the k nearest entries of an external memory (an Index), softmax-weighted."""
    title = "Retrieval (attention over a memory)"
    k: int

    def __call__(self) -> Model[tuple[Container, Index], Container]: ...


class Mixup(Aggregate[Container, tuple[Container, Value]]):
    """Each example blended with a random partner of the batch, labels by the same λ: the batch as a Set, paired at
    random."""
    title = "Data augmentation: mixup"
    side = "agnostic"
    alpha: float = 0.2

    def __call__(self, x: Container, y: Value, /) -> tuple[Container, Value]: ...


class Cutmix(Aggregate[Grid, tuple[Grid, Value]]):
    """A random box pasted from a partner image; the labels mix by its area."""
    title = "Data augmentation: cutmix"
    side = "agnostic"

    def __call__(self, x: Grid, y: Value, /) -> tuple[Grid, Value]: ...


class BatchBaseline(Aggregate[Value, Value]):
    """Each return minus the batch's mean: an advantage with less variance and no bias."""
    title = "Baseline: the batch mean"
    side = "agnostic"

    def __call__(self, returns: Value, /) -> Value: ...


class GroupRelative(Aggregate[Value, Value]):
    """Each reward standardized within its group of n samples from one prompt: an advantage without a critic."""
    title = "GRPO (group-relative advantages)"
    side = "agnostic"
    n: int

    def __call__(self, rewards: Value, /) -> Value: ...


# ═══ Select[A, B]: C[a] → C[a] ════════════════════════════════════════════════════════════════════


class QualityFilter(Select[Container, Container]):
    """Keep the examples of a batch scoring above a threshold (a classifier's, a verifier's, a perplexity). Keeping
    the best fraction needs every batch: KeepBest."""
    title = "Data filtering"
    threshold: float

    def __call__[C: Container](self, x: C, scores: Value, /) -> C: ...


class TokenDropout(Select[Seq, Seq]):
    """Drop a random fraction of the positions; the rest keep their order."""
    title = "Data augmentation: token dropout"
    p: float = 0.1

    def __call__(self, x: Seq, /) -> Seq: ...


class DecayGroups(Select[nn.Module, list[dict[str, Any]]]):
    """The parameters to decay (the weight matrices) apart from the rest (norms, biases, embeddings)."""
    title = "Weight decay (parameter groups)"
    weight_decay: float = 0.1

    def __call__(self, model: nn.Module, /) -> list[dict[str, Any]]: ...


class MuPGroups(Select[nn.Module, list[dict[str, Any]]]):
    """Hidden matrices reweighted by base_width / width, so a learning rate tuned narrow carries over wide."""
    title = "Parameterization (μP)"
    width: int
    base_width: int

    def __call__(self, model: nn.Module, /) -> list[dict[str, Any]]: ...


# ═══ Fold[A, B]: C[a] → a ═════════════════════════════════════════════════════════════════════════


class PoolFor(Fold[Container, Value]):
    """A container's features → one vector per example: a causal Seq keeps its last position, the one that read
    everything; anything else, the mean."""
    title = "Pooling (mean / last position)"

    def __call__(self, x: Container, /) -> Model[Container, Value]: ...


class DiscountedReturn(Fold[Path, Value]):
    """Σ γ^t·r_t along each path: what RL maximizes. A walk can't be differentiated through, so its gradient comes
    from scoring the path (reinforce)."""
    title = "Expected return (the RL objective)"
    gamma: float = 1.0

    def __call__(self, path: Path, /) -> Value: ...


class Best(Fold[Path, Path]):
    """The best-scoring of each group of n paths (scored by a verifier or a reward model)."""
    title = "Best-of-N with a verifier"
    n: int

    def __call__(self, paths: Path, scores: Value, /) -> Path: ...


class Majority(Fold[Value, Value]):
    """The most frequent answer of each group of n samples."""
    title = "Self-consistency (majority vote)"
    n: int

    def __call__(self, answers: Value, /) -> Value: ...


class EarlyStop(Fold[Seq, bool]):
    """Whether the validation loss has stopped improving for `patience` evaluations."""
    title = "Early stopping"
    patience: int = 3

    def __call__(self, curve: Seq, /) -> bool: ...


class FitPowerLaw(Fold[Seq, tuple[float, float]]):
    """loss ≈ a·size^(−b), fitted on the logs."""
    title = "Scaling laws (power-law fit)"

    def __call__(self, sizes: Seq, losses: Seq, /) -> tuple[float, float]: ...


# ═══ Scan[X, S, Y]: Seq[X] → Seq[Y] ═══════════════════════════════════════════════════════════════


class Optimizer(Scan[Tensor, Tensor, Tensor]):
    """Gradients in, updates out: SGD carries nothing, momentum a velocity, Adam two moments, Muon orthogonalizes.
    A setting of Train, which carries its state from step to step."""
    title = "Optimizers (SGD, momentum, Adam, Muon)"
    state = "the moments of every parameter"
    kind: str = "adamw"
    lr: float = 3e-4


class WarmupCosine(Scan[int, int, float]):
    """A learning rate for each step: linear warmup, then cosine decay. A setting of Train."""
    title = "Learning-rate schedules"
    state = "the step count"
    warmup: int
    total: int


class Lagrangian(Scan[tuple[Tensor, Tensor], float, Tensor]):
    """(loss f, constraint g) in, f + λ·(g − c) out; λ rises while the constraint is violated (dual ascent)."""
    title = "Constrained objectives (Lagrangian)"
    state = "the multiplier λ ≥ 0"
    lr: float = 0.1


class ReturnsToGo(Scan[Tensor, Tensor, Tensor]):
    """G_t = r_t + γ·G_{t+1}: what each step led to, scanned from the end."""
    title = "Returns-to-go"
    state = "the return from here on"
    reverse = True
    gamma: float = 1.0


class Negamax(Scan[Tensor, Tensor, Tensor]):
    """One policy plays both sides; each move's return is the outcome for whoever made it: G_t = r_t − G_{t+1}."""
    title = "Self-play"
    state = "the outcome for the player to move"
    reverse = True


class GAE(Scan[tuple[Tensor, Tensor], Tensor, Tensor]):
    """A critic's TD errors r + γV(s′) − V(s), averaged backward with weights (γλ)^k; each element a (reward, value)."""
    title = "Value functions, Bellman, GAE"
    state = "the advantage from here on"
    reverse = True
    gamma: float = 0.99
    lmbda: float = 0.95


class ParallelScan(Scan[Tensor, Tensor, Tensor]):
    """s_t = a_t·s_{t−1} + b_t for every t at once, in logarithmic depth: how a recurrence trains in parallel."""
    title = "Parallel scan"
    function = "torch.cumsum"


# ═══ Split[A]: C → Example ════════════════════════════════════════════════════════════════════════


class SplitFor(Split[Container]):
    """The split a container's type picks: classes in order → Shift · other classes → Mask · real values → Noise."""
    title = "Choosing the split from the data"

    def __call__(self, x: Container, /) -> Example: ...


class Labelled(Split[Container]):
    """The data comes split: the input with nothing hidden, the label as the target."""
    title = "Supervised"

    def __call__(self, x: Container, y: Container, /) -> Example: ...


class Shift(Split[Seq]):
    """Each element is predicted from the ones before it (a causal Seq): the target at i is the element at i + 1."""
    title = "Self-supervised: next-token prediction"

    def __call__(self, x: Seq, /) -> Example: ...


class Mask(Split[Container]):
    """Random elements at level 1 ([MASK] for classes, 0 for real values): 15% is BERT, 75% MAE, a random ratio per
    example MaskGIT and masked diffusion."""
    title = "Self-supervised: masked prediction (BERT, MAE, MaskGIT)"
    ratio: float | None = None

    def __call__(self, x: Container, /) -> Example: ...


class Noise(Split[Container]):
    """Every element at one random level t (x_t = (1 − t)·x + t·ε): the target is the velocity, the noise or the
    clean data, rescalings of one another."""
    title = "Diffusion / score matching / flow matching"

    def __call__(self, x: Container, /) -> Example: ...


class NoiseEach(Split[Seq]):
    """A random level for each element: next-token prediction and diffusion are its two extremes."""
    title = "Diffusion forcing"

    def __call__(self, x: Seq, /) -> Example: ...


class NextFrame(Split[Product]):
    """The frames up to t as the input, the frames from t + 1 as the target: shift, on the frames of a video."""
    title = "Self-supervised: future-frame prediction"

    def __call__(self, x: Product, /) -> Example: ...


class Reconstruction(Split[Container]):
    """Nothing hidden; the target is the input itself, so the model's bottleneck is what makes it hard."""
    title = "Reconstruction (autoencoder)"

    def __call__(self, x: Container, /) -> Example: ...


class TwoViews(Split[Container]):
    """Two random augmentations of the same examples, each the other's target (CLIP pairs an image with its caption)."""
    title = "Contrastive views"

    def __call__(self, x: Container, augment: Callable[[Container], Container], /) -> Example: ...


class Distillation(Split[Container]):
    """The target is a teacher's (softened) prediction on the same input."""
    title = "Knowledge distillation"

    def __call__(self, x: Container, teacher: Predictor, /) -> Example: ...


class SelfDistillation(Split[Container]):
    """One view as the input; the target is a slow (EMA) copy's embedding of another view: agree with an older self."""
    title = "Self-distillation (BYOL / DINO / JEPA)"

    def __call__(self, x: Container, teacher: Predictor, augment: Callable[[Container], Container], /) -> Example: ...


class SFT(Split[Seq]):
    """shift, with only the response as targets: the prompt is read, never predicted."""
    title = "Instruction tuning / SFT"
    prompt_len: int

    def __call__(self, x: Seq, /) -> Example: ...


class TDTarget(Split[Value]):
    """V(s) learns toward r + γ·V(s′): a target bootstrapped from the critic's own estimate."""
    title = "TD learning (value regression)"
    gamma: float = 0.99

    def __call__(self, batch: Value, critic: Predictor, /) -> Example: ...


class QTarget(Split[Value]):
    """Q(s, a) learns toward r + γ·max Q(s′, ·) at the action taken (the weight picks it out)."""
    title = "Q-learning (DQN)"
    gamma: float = 0.99

    def __call__(self, batch: Value, q: Predictor, /) -> Example: ...


class Dynamics(Split[Value]):
    """From each transition, the input is the state and the action; the target what followed (next state, reward,
    done). A model of it is a learned follow (imagined)."""
    title = "World-model targets (dynamics)"

    def __call__(self, batch: Value, /) -> Example: ...


class SearchTargets(Split[Path]):
    """For each state a walk visited: what search made of it (its visit distribution) and what followed (the return)."""
    title = "Amortized search (AlphaZero targets)"

    def __call__(self, path: Path, search: PartialTree[Any, Any], /) -> Example: ...


# ═══ Score[A, B]: Prediction × C → ℝ ══════════════════════════════════════════════════════════════


class Objective(Score[nn.Module, Tensor]):
    """The loss of a model on a batch: split it into an Example, predict from the input, score with the loss the
    target picks (labelled data: split = Labelled())."""
    title = "Objective (a split, then its loss)"
    split: Split[Any]

    def __call__(self, model: nn.Module, /, *batch: Any) -> Tensor: ...


class Loss(Score[Prediction, Tensor]):
    """−log p(target), chosen by what the target holds: classes → cross-entropy · bits → BCE · real → squared error ·
    distribution → KL · record → the sum of its fields' losses."""
    title = "Loss = negative log-likelihood of the target"

    def __call__(self, pred: Prediction, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


class CrossEntropy(Score[Tensor, Tensor]):
    """Classes, or a distribution over them (soft labels)."""
    title = "Cross-entropy"

    def __call__(self, logits: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


class BinaryCrossEntropy(Score[Tensor, Tensor]):
    """Bits."""
    title = "Binary cross-entropy"

    def __call__(self, logits: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


class SquaredError(Score[Tensor, Tensor]):
    """Real values: a unit Gaussian's negative log-likelihood, up to a constant."""
    title = "Squared error"

    def __call__(self, pred: Tensor, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


class KL(Score[Dist, Tensor]):
    """A distribution against a reference (a prior, a reference policy)."""
    title = "KL divergence"

    def __call__(self, pred: Dist, target: Dist, weight: Tensor | None = None, /) -> Tensor: ...


class PerField(Score[Prediction, Tensor]):
    """A record: one loss per field, each chosen by what the field holds, summed."""
    title = "Multi-task training"

    def __call__(self, pred: Prediction, target: Container, weight: Tensor | None = None, /) -> Tensor: ...


class LogProb(Score[Prediction, Value]):
    """log p(x) for each element under a prediction, unreduced: what losses and estimators are built from."""
    title = "Log-probability"

    def __call__(self, pred: Prediction, x: Container, /) -> Value: ...


class PathLogProb(Score[PartialTree[Any, Any], Value]):
    """Σ log options(node)(edge) along a path; on the data's own path, −1/length of it is the next-token loss."""
    title = "Teacher forcing (the log-probability of a path)"

    def __call__(self, tree: PartialTree[Any, Any], path: Path, /) -> Value: ...


class SequenceLogProb(Score[Predictor, Value]):
    """log p(x[start:] | the rest before it), per sequence: teacher forcing on whole sequences, in one pass."""
    title = "Sequence log-probability"
    start: int = 1

    def __call__(self, model: Predictor, x: Seq, /) -> Value: ...


class InfoNCE(Score[Value, Tensor]):
    """Which b goes with each a, among the batch: cross-entropy over similarities, the matching pair as the class."""
    title = "Contrastive (InfoNCE / CLIP)"
    temperature: float = 0.07

    def __call__(self, a: Value, b: Value, /) -> Tensor: ...


class BradleyTerry(Score[Value, Tensor]):
    """−log σ(r(chosen) − r(rejected)): the likelihood that the preferred answer wins."""
    title = "Reward models (Bradley–Terry)"

    def __call__(self, chosen: Value, rejected: Value, /) -> Tensor: ...


class DPOLoss(Score[nn.Module, Tensor]):
    """The loss of (policy, chosen, rejected): Bradley–Terry on β·(the policy's log-ratio to the reference, chosen
    minus rejected). Preferences as classification: no reward model, no walk."""
    title = "Preferences (DPO)"
    reference: Predictor
    beta: float = 0.1

    def __call__(self, model: nn.Module, /, *batch: Any) -> Tensor: ...


class KLRegularizedReward(Score[Value, Value]):
    """The reward, minus how far the policy has moved from a reference: use it as the return."""
    title = "KL-regularized reward (RLHF)"
    beta: float = 0.1

    def __call__(self, reward: Value, LogProb: Value, ref_log_prob: Value, /) -> Value: ...


class ELBO(Score[Model[Container, Dist], Tensor]):
    """The reconstruction from a latent drawn from the posterior (rsample: reparameterization), plus KL(posterior ‖ prior)."""
    title = "VAE (evidence lower bound)"

    def __call__(self, encoder: Model[Container, Dist], decoder: Model[Tensor, Prediction], x: Container, /) -> Tensor: ...


class GAN(Score[Model[Tensor, Container], tuple[Tensor, Tensor]]):
    """The discriminator tells real from generated; the generator wants its samples called real. Two scores, for two
    players (alternate their steps)."""
    title = "GAN (two players)"

    def __call__(self, generator: Model[Tensor, Container], discriminator: Predictor, real: Container, z: Tensor, /) -> tuple[Tensor, Tensor]: ...


class FlowLogProb(Score[Container, Value]):
    """The exact log-likelihood through invertible maps (change of variables)."""
    title = "Normalizing flows"
    side = "model"

    def __call__(self, x: Container, transforms: list[Any], /) -> Value: ...


class Reinforce(Score[Path, Tensor]):
    """−E[log p(path)·A]: a walked path's log-probability, weighted by how much better than usual it went."""
    title = "Policy gradients (REINFORCE)"

    def __call__(self, path: Path, advantage: Value, /) -> Tensor: ...


class PPOClip(Score[Value, Tensor]):
    """The policy gradient with the ratio π / π_old clipped: it stops pushing once the policy has moved far."""
    title = "PPO clipped surrogate"
    clip: float = 0.2

    def __call__(self, LogProb: Value, old_log_prob: Value, advantage: Value, /) -> Tensor: ...


class Pathwise(Score[Dist, Tensor]):
    """−E[Q(a)] with a = rsample(): differentiated through the sample into a learned critic (DDPG, SAC)."""
    title = "Pathwise estimator (through a critic)"

    def __call__(self, policy: Dist, critic: Model[Tensor, Tensor], /) -> Tensor: ...


class EntropyBonus(Score[Dist, Tensor]):
    """−α·H(options): keeps the policy exploring."""
    title = "Maximum-entropy RL (entropy bonus)"
    alpha: float = 0.01

    def __call__(self, options: Dist, /) -> Tensor: ...


class MAMLLoss(Score[nn.Module, Tensor]):
    """Bilevel: the query loss of the weights after an inner step (an unroll) on the support set, differentiated through."""
    title = "Meta-learning (MAML)"

    def __call__(self, model: nn.Module, objective: Callable[..., Tensor], support: Container, query: Container, /) -> Tensor: ...


class Accuracy(Score[Tensor, Tensor]):
    """Reported and selected on, never differentiated: a loss stands in for it in training."""
    title = "Metrics (accuracy)"

    def __call__(self, pred: Tensor, target: Container, /) -> Tensor: ...


class RetrievalAccuracy(Score[Value, Tensor]):
    """How often each a is closest to its own b among the batch (matching images to captions)."""
    title = "Retrieval accuracy"

    def __call__(self, a: Value, b: Value, /) -> Tensor: ...


class Autodiff(Score[Tensor, Tensor]):
    """The gradient of any score."""
    title = "Backpropagation / autodiff"
    function, signature = "torch.autograd", "ℝ → ∇"


# ═══ Start[P]: … → Partial ════════════════════════════════════════════════════════════════════════


class Blank(Start[Holes[Container]]):
    """Every element unknown: all [MASK] for classes, pure noise for real values."""
    title = "Blank (where generation starts)"

    def __call__[C: Container](self, like: C, /) -> Holes[C]: ...


class LMTree(Start[PartialTree[Seq, Tensor]]):
    """A node is a sequence (Holes: known up to its end), its options the next-token distribution, an edge appends."""
    title = "An LM as a tree"

    def __call__(self, model: Predictor, prompt: Seq, /) -> PartialTree[Seq, Tensor]: ...


class EnvTree(Start[PartialTree[Any, Tensor]]):
    """A node is a state, its options the policy's actions (uniform with none), an edge leads where the environment
    steps to, with a reward and whether it ended."""
    title = "MDP (an environment as a tree)"

    def __call__(self, env: Environment, policy: Predictor | None = None, /) -> PartialTree[Any, Tensor]: ...


# ═══ Order: Holes × Prediction → Holes ════════════════════════════════════════════════════════════


class InOrder(Order):
    """Fill the first hole: one element at a time, in order (the holes view of decoding; walk gives the tree view)."""
    title = "Autoregressive order"
    state = "Holes[C] — known up to i"


class MostConfident(Order):
    """Fill the holes the model is surest of, more each step (a cosine schedule), until none is left."""
    title = "Unmasking (MaskGIT, masked diffusion)"
    state = "Holes[C] — some holes at 1, the rest known"


class Together(Order):
    """Lower every level by one step along the predicted velocity (Euler): noise, undone."""
    title = "Sampling a flow (diffusion)"
    state = "Holes[C] — every element at level t"


class PerElement(Order):
    """Each element follows its own schedule (the near future clean, the far future noisy): between InOrder and together."""
    title = "Diffusion forcing sampling"
    state = "Holes[C] — a level for each element"


# ═══ Walk: PartialTree → Path ═════════════════════════════════════════════════════════════════════


class ChainOfThought(Walk):
    """A longer walk before the answer; the gains come from training the walk (R1)."""
    title = "Chain-of-thought / longer reasoning"
    function = "Walk"


# ═══ Choose: PartialTree → PartialTree ════════════════════════════════════════════════════════════


class Sample(Choose):
    """Expand each kept leaf by one edge drawn from its options."""
    title = "Sampling"
    state = "PartialTree — one leaf per root"


class Greedy(Choose):
    """Expand each kept leaf by its likeliest edge."""
    title = "Greedy decoding"
    state = "PartialTree — one leaf per root"


class Beam(Choose):
    """Expand every kept leaf; keep the k likeliest children of each root."""
    title = "Beam search"
    state = "k leaves per root and their log-probabilities"
    k: int


class Independent(Choose):
    """n paths from each root side by side, each grown by its own chooser: n samples per prompt (best-of-N, GRPO)."""
    title = "Several paths per root (best-of-N, groups)"
    state = "n independent leaves per root"
    n: int
    choose: Chooser


class Forced(Choose):
    """Expand each leaf by the next given edge: scoring the path walked is teacher forcing."""
    title = "Teacher forcing (walking the data's path)"
    state = "one leaf per root, on the data's path"
    edges: Seq


class Speculative(Choose):
    """The draft walks k edges ahead greedily, the tree's model checks them in one pass, the agreeing prefix is kept:
    greedy decoding, in fewer of the big model's passes."""
    title = "Speculative decoding"
    state = "the accepted leaf, and a draft model's guesses past it"
    draft: PartialTree[Any, Any]
    k: int = 4


# ═══ Unroll: PartialSeq[S] → S ════════════════════════════════════════════════════════════════════


class Iteration(Unroll):
    """Rounds of collecting and learning (policy iteration, self-play, expert iteration): a script's repeat
    (program.py), since each round's fit is a statement."""
    title = "Iteration (rounds of a program)"
    function = "repeat"


# ═══ Evolve[S]: S → S ═════════════════════════════════════════════════════════════════════════════


class Langevin(Evolve[Container]):
    """x ← x − η·∇E(x) + √(2η)·ε: samples of exp(−energy), with the gradient from autograd."""
    title = "Energy-based models (Langevin)"
    state = "a sample, moving"
    energy: Predictor
    step_size: float = 0.01


class EvolutionStrategy(Evolve[Dist]):
    """Draw a population (a Set of parameters) from the search distribution, score them, move it toward the better
    ones: it works when nothing is differentiable (CMA-ES keeps a covariance too)."""
    title = "Evolution strategies (no gradient)"
    state = "a search distribution over the parameters"
    fitness: Callable[[Tensor], Tensor]
    population: int = 32


# ═══ Reshape[A, B]: Partial → Partial ═════════════════════════════════════════════════════════════


class Temperature(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """The options' logits divided by t: below 1 sharper (toward greedy), above 1 flatter."""
    title = "Sampling temperature"
    t: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class TopK(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """Only the k likeliest edges stay possible."""
    title = "Top-k sampling"
    k: int

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class TopP(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """Only the likeliest edges whose probabilities add up to p stay possible."""
    title = "Top-p (nucleus) sampling"
    p: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class Given(Reshape[Predictor, Prediction]):
    """A conditional model with its condition fixed (None: no condition): the two models guidance mixes."""
    title = "A condition, fixed"

    def __call__(self, model: Predictor, condition: Value | None, /) -> Predictor: ...


class Guided(Reshape[Predictor, Prediction]):
    """u + w·(c − u): extrapolate from the unconditional prediction toward the conditional one, for a tree's options or
    a fill's model."""
    title = "Classifier-free guidance"
    w: float

    def __call__(self, cond: Predictor, uncond: Predictor, /) -> Predictor: ...


class EpsilonGreedy(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """With probability ε a uniform edge, else the likeliest."""
    title = "Exploration vs exploitation (ε-greedy)"
    eps: float

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class WithTool(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """When a walk emits the call token, follow writes the tool's answer (a calculator, a search, an interpreter)."""
    title = "Tool use"
    call: int

    def __call__(self, tree: PartialTree[Any, Any], tool: Callable[[Any], Tensor], /) -> PartialTree[Any, Any]: ...


class Cached(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """The tree's known nodes keep what the model computed for them, so expanding a node costs one position."""
    title = "KV caching"
    state = "each known node keeps its keys and values"

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class Imagined(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """The environment's follow (and reward, done) replaced by a learned model of them: walks, searches and plans in
    imagination (Dreamer, MuZero)."""
    title = "World models (a learned follow)"

    def __call__(self, tree: PartialTree[Any, Any], world: Model[Container, Container], /) -> PartialTree[Any, Any]: ...


class Planned(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """Options that look ahead: random action sequences simulated in a world, all the probability on the first action
    of the best."""
    title = "Planning (random shooting / MPC)"
    state = "the random plans tried from each node"
    horizon: int = 5
    samples: int = 64

    def __call__(self, tree: PartialTree[Any, Any], world: PartialTree[Any, Any] | None = None, /) -> PartialTree[Any, Any]: ...


class MCTS(Reshape[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """Options improved by search: grow a tree by PUCT (the options as priors), back up rewards and values, return the
    visit counts. Walk it to search; train toward it to amortize the search (AlphaZero)."""
    title = "MCTS (tree search)"
    state = "a search tree per node: priors, visit counts, values"
    simulations: int = 64
    two_player: bool = False

    def __call__(self, tree: PartialTree[Any, Any], /) -> PartialTree[Any, Any]: ...


class Droppable(Reshape[Predictor, Prediction]):
    """The condition dropped with probability p while training, so one model is both conditional and unconditional."""
    title = "Classifier-free guidance training"
    p: float = 0.1

    def __call__(self, model: Predictor, /) -> Predictor: ...


# ═══ Search[A, B]: Index × Value → Set ════════════════════════════════════════════════════════════


class EmptyIndex(Search[Value, Index]):
    """An empty index for vectors like these (their size): flat (exact), IVF (clusters), HNSW (a neighbour graph), PQ
    (compressed codes), LSH (hash buckets), inverted (BM25), sum tree (by priority). Statements train and fill it."""
    title = "Index (an empty one)"
    kind: str = "flat"

    def __call__(self, like: Value, /) -> Index: ...


class Retrieve(Search[Index, Seq]):
    """The k documents nearest the query's embedding, put in front of it as one prompt (then walk the LM's tree)."""
    title = "Retrieval-augmented generation (retrieve)"
    k: int = 4

    def __call__(self, index: Index, documents: Set, query: Seq, embed: Predictor, /) -> Seq: ...


class kNNOptions(Search[PartialTree[Any, Any], PartialTree[Any, Any]]):
    """Options mixed with the next tokens of the k nearest stored contexts: an LM that can look things up."""
    title = "kNN-LM (a memory of contexts)"
    weight: float = 0.25

    def __call__(self, tree: PartialTree[Any, Any], memory: Index, embed: Predictor, /) -> PartialTree[Any, Any]: ...


class HardNegatives(Search[Index, Set]):
    """The nearest elements that are not the match: harder negatives for a contrastive loss."""
    title = "Hard-negative mining"
    k: int = 8

    def __call__(self, index: Index, queries: Value, positives: Value, /) -> Set: ...


# ═══ Wrap[A, B]: Module → Module′ ═════════════════════════════════════════════════════════════════


class Residual(Wrap[nn.Module, nn.Module]):
    """h + f(norm h): the layer edits the stream instead of replacing it."""
    title = "Residual connections"
    side = "model"

    def __call__(self, f: nn.Module, /) -> nn.Module: ...


class FrozenCopy(Wrap[nn.Module, nn.Module]):
    """A copy that no longer trains: DPO's and RLHF's reference, DQN's target network."""
    title = "Frozen copy (reference / target network)"

    def __call__[M: nn.Module](self, model: M, /) -> M: ...


class Observe(Wrap[nn.Module, nn.Module]):
    """A model that returns what one of its modules computed: the activations probes, SAEs and circuits study."""
    title = "Reading activations"

    def __call__(self, model: nn.Module, at: nn.Module, /) -> nn.Module: ...


class Patch(Wrap[nn.Module, nn.Module]):
    """A model with one module's output replaced (by another input's activations): which components carry a behaviour?"""
    title = "Activation patching (circuits)"

    def __call__(self, model: nn.Module, at: nn.Module, replace: Callable[[Tensor], Tensor], /) -> nn.Module: ...


class Steer(Wrap[nn.Module, nn.Module]):
    """α·direction added to one module's output: features are directions, and moving along one changes the behaviour."""
    title = "Steering along a direction"
    alpha: float = 1.0

    def __call__(self, model: nn.Module, at: nn.Module, direction: Tensor, /) -> nn.Module: ...


class Autocast(Wrap[nn.Module, nn.Module]):
    """bf16 (or fp8) compute, float32 master weights."""
    title = "Low precision: training"
    function = "torch.autocast"


class Compile(Wrap[nn.Module, nn.Module]):
    """torch.compile(module) fuses its ops; attention picks flash kernels by itself."""
    title = "Hardware-aware kernels"
    side = "model"
    function = "torch.compile"


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# STATEMENTS · on a dataset, or changing something in place (!)
# ══════════════════════════════════════════════════════════════════════════════════════════════════

# ═══ Fit[X]: X! × Dataset ═════════════════════════════════════════════════════════════════════════


class Train(Fit[nn.Module]):
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


class FitCodec(Fit[Codec[Any, Any]]):
    """Fit a codec to a corpus by its own rule, in place: BPE merges the most frequent pairs until the vocabulary is
    full, a unigram model prunes. (A learned codec, VQ-VAE or EnCodec, is a model: train.)"""
    title = "Codec fitting (BPE merges, a vocabulary)"

    def __call__(self, codec: Codec[Any, Any], data: Dataset[Any], /) -> None: ...


class TrainingStep(Fit[nn.Module]):
    """Train's settings: autocast the objective, clip the gradient norm."""
    title = "Training step (clipping, mixed precision)"
    function = "Train"


class TwoPlayerGames(Fit[nn.Module]):
    """No single quantity goes down: each player's Train on its own loss, in turn, inside a repeat (GANs,
    adversarial training)."""
    title = "Two-player games (alternating fits)"
    function = "repeat, Train"


# ═══ Assign[X]: Weights! ══════════════════════════════════════════════════════════════════════════


class Transfer(Assign[Weights]):
    """Start a model from a trained one: copy into it the weights whose names and shapes match (the body), keep its head."""
    title = "Pretrain → finetune (transfer)"
    changes = 1

    def __call__(self, source: Weights, target: Weights, /) -> None: ...


class Init(Assign[Weights]):
    """Fan-in normal for matrices, zeros for output projections: each residual block starts as the identity."""
    title = "Initialization"

    def __call__(self, model: Weights, /) -> None: ...


class EMA(Assign[Weights]):
    """average ← decay·average + (1 − decay)·model, in place: after each round or step, a smoother copy, and the slow
    target of self-distillation and DQN."""
    title = "EMA of weights"
    decay: float = 0.999

    def __call__(self, average: Weights, model: Weights, /) -> None: ...


# ═══ Alter[X]: Module! ════════════════════════════════════════════════════════════════════════════


class Freeze(Alter[nn.Module]):
    """Train only the parameters whose name passes a test."""
    title = "Freezing"
    side = "agnostic"

    def __call__(self, model: nn.Module, trainable: Callable[[str], bool], /) -> None: ...


class LoRA(Alter[nn.Module]):
    """W + B·A on chosen linear layers, the base frozen: the finetuning update, kept low-rank."""
    title = "LoRA"
    rank: int = 8

    def __call__(self, model: nn.Module, /) -> None: ...


class Adapter(Alter[nn.Module]):
    """After chosen frozen layers, a small bottleneck MLP that trains alone, starting at zero."""
    title = "Adapters"
    bottleneck: int = 8

    def __call__(self, model: nn.Module, /) -> None: ...


class PromptTuning(Alter[nn.Module]):
    """Learned vectors in front of a frozen model's input; only they train."""
    title = "Prompt tuning"
    n: int

    def __call__(self, model: nn.Module, /) -> None: ...


class Quantize(Alter[Weights]):
    """Weights rounded to 2^(bits−1) − 1 levels a side per row: computing as the low-precision model would."""
    title = "Quantization (int8 / int4)"
    bits: int = 8

    def __call__(self, model: Weights, /) -> None: ...


class LowPrecisionInference(Alter[Weights]):
    """int8 / int4 weights."""
    title = "Low precision: inference"
    function = "Quantize"


class Parallelism(Alter[nn.Module]):
    """Data, sharded, tensor, pipeline, expert, sequence: fully_shard shards a module in place; DTensor places its
    weights."""
    title = "Parallelism"
    side = "agnostic"
    function = "torch.distributed"


# ═══ Store[X]: Index! / Buffer! ═══════════════════════════════════════════════════════════════════


class TrainIndex(Store[Index]):
    """Learn the index's structure from a sample of the vectors, in place: k-means centroids for IVF, codebooks for
    PQ (flat and HNSW have nothing to learn). CREATE INDEX, before any rows."""
    title = "Index training (IVF centroids, PQ codebooks)"

    def __call__(self, index: Index, sample: Value, /) -> None: ...


class Insert(Store[Index]):
    """Add every batch of the dataset to the index, in place (FAISS add, INSERT). An index over an expression's output
    (embed(x)) goes stale when that expression's model is trained again: build it again."""
    title = "Index insertion"

    def __call__(self, index: Index, data: Dataset[Set], /) -> None: ...


class Delete(Store[Index]):
    """Remove elements from the index by id, in place (FAISS remove_ids, DELETE)."""
    title = "Index deletion"

    def __call__(self, index: Index, ids: Tensor, /) -> None: ...


class Extend(Store[Buffer[Container]]):
    """Add a batch to a bounded buffer, in place, dropping the oldest (or the lowest priority): past experience to train
    on again (continual learning, off-policy RL, prioritized replay)."""
    title = "Continual learning (replay)"

    def __call__[C: Container](self, buffer: Buffer[C], batch: C, /) -> None: ...


# ═══ Source[D]: … → Dataset ═══════════════════════════════════════════════════════════════════════


class Batches(Source[Dataset[Any]]):
    """Containers in memory as a dataset: random minibatches of size examples, the containers sliced together (a
    tuple per batch when there are several)."""
    title = "Minibatches"
    size: int
    shuffle: bool = True

    def __call__(self, *data: Container) -> Dataset[Any]: ...


class Collector(Source[Dataset[Path]]):
    """Walks of the environment with the policy as it is when each batch is drawn: on-policy data, read inside the
    fit that changes the policy (torchrl's collectors)."""
    title = "Collector (walks of a live policy)"
    steps: int
    batch: int

    def __call__(self, env: Environment, policy: Predictor, /) -> Dataset[Path]: ...


class ReplayBuffer(Source[Buffer[Any]]):
    """An empty buffer of at most size examples, read by sampling (uniformly, or by priority); Extend fills it."""
    title = "Replay buffer (empty)"
    size: int
    priority: bool = False

    def __call__(self) -> Buffer[Any]: ...


class Mix(Source[Dataset[Any]]):
    """Batches drawn from several datasets in proportion to their weights (web, code, books): the mixture as a dataset."""
    title = "Data mixture weights"
    weights: list[float]

    def __call__(self, *datasets: Dataset[Any]) -> Dataset[Any]: ...


class Each(Source[Dataset[Any]]):
    """f applied to every batch, once per batch as the dataset batches it (never re-batched, since f may depend on
    what a batch holds): how a statement reads an expression's output, embeddings to index, features to probe."""
    title = "An expression on each batch"

    def __call__(self, f: Callable[..., Any], data: Dataset[Any], /) -> Dataset[Any]: ...


# ═══ Curate[C]: Dataset → Dataset ═════════════════════════════════════════════════════════════════


class Dedup(Curate[Container]):
    """Drop exact copies across the whole dataset (by hash): the first of each kept."""
    title = "Data deduplication"

    def __call__[C: Container](self, data: Dataset[C], /) -> Dataset[C]: ...


class DropNearDuplicates(Curate[Container]):
    """Drop examples whose MinHash signatures collide with an earlier one's above a similarity threshold: dedup beyond
    exact copies."""
    title = "Near-duplicate removal (MinHash / LSH)"
    threshold: float = 0.8

    def __call__[C: Container](self, data: Dataset[C], /) -> Dataset[C]: ...


class KeepBest(Curate[Container]):
    """The best-scoring fraction of the whole dataset: a quantile of every score first, then a threshold per batch."""
    title = "Data filtering by fraction"
    keep: float = 0.5

    def __call__[C: Container](self, data: Dataset[C], score: Callable[[C], Value], /) -> Dataset[C]: ...


class Curriculum(Curate[Container]):
    """The dataset sorted by difficulty, easy first: a global sort, so it reads every batch before the first comes out."""
    title = "Curriculum learning"

    def __call__[C: Container](self, data: Dataset[C], difficulty: Callable[[C], Value], /) -> Dataset[C]: ...


# ═══ Reduce[A, B]: Dataset → a ════════════════════════════════════════════════════════════════════


class Evaluate(Reduce[Dataset[Any], Tensor]):
    """A per-batch metric (accuracy, a loss, perplexity), averaged over every batch, weighted by their sizes."""
    title = "Evaluation (a metric over a dataset)"

    def __call__(self, metric: Callable[..., Tensor], data: Dataset[Any], /) -> Tensor: ...


class Accumulate(Reduce[Dataset[Any], Any]):
    """A Scan's state carried across a dataset's batches, its final state returned: normalization statistics, a
    histogram, any metric that is not a mean."""
    title = "Accumulate (a scan across batches)"

    def __call__(self, s: Scan[Any, Any, Any], data: Dataset[Any], /) -> Any: ...


class Peek(Reduce[Dataset[Any], Any]):
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
       "State-space models / linear recurrences", "Walk"}


def _paint(code: str) -> Callable[[str], str]:
    """An ANSI colour, or nothing when the output isn't a terminal (or NO_COLOR is set)."""
    import os
    import sys

    on = (sys.stdout.isatty() or os.environ.get("FORCE_COLOR")) and not os.environ.get("NO_COLOR")
    return (lambda t: f"\033[{code}m{t}\033[0m") if on else (lambda t: t)


def _about(g: type[Method]) -> str:
    return g.__dict__.get("about") or " ".join((inspect.getdoc(g) or "").split("\n\n")[0].split())


def print_catalog(explain: bool = False) -> None:
    """Grouped by framework, family and group, one line per method: name: signature · state, then its settings.
    explain adds what each one does."""
    bold, group, sig, state_c, dim, star = _paint("1"), _paint("1;36"), _paint("33"), _paint("35"), _paint("2"), _paint("32")
    for heading, families in FRAMEWORKS.values():
        print(bold(f"\n━━━━ {heading} ━━━━"))
        for family in families:
            print(bold(f"\n══ {FAMILIES[family]} ══"))
            for g in (g for g in GROUPS.values() if g.family == family):
                print(f"\n{group(g.__name__)}  {sig(g.form)}\n  {dim(_about(g))}")
                for e in (e for e in ENTRIES if e.group == g.group):
                    mark = star("★ ") if e.name in NEW else ""
                    state = f"  {dim('·')} {state_c('state: ' + e.state)}" if e.state else ""
                    settings = dim(f"  [{', '.join(e.settings)}]") if e.settings and e.cls is not None and not e.cls.function else ""
                    print(f"  - {mark}{bold(e.name)}: {sig(e.signature)}{state}{settings}")
                    if explain:
                        print(dim(f"      {e.function}: {e.explanation}"))
    print(bold("\n══ META-PRIMITIVES · properties of the groups ══\n"))
    for e in (e for e in ENTRIES if e.group == "meta"):
        print(f"  - {bold(e.name)}: {group(e.signature)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    print(bold("\n══ PHENOMENA · observed, not computed ══\n"))
    for e in (e for e in ENTRIES if e.group == "phenomenon"):
        print(f"  - {bold(e.name)}" + (dim(f"\n      {e.explanation}") if explain else ""))
    rows = [e for e in ENTRIES if e.group in GROUPS]
    stmts = [e for e in rows if issubclass(GROUPS[e.group], Stmt)]
    print(dim(f"\n{len(rows)} methods in {len(GROUPS)} groups, {sum(bool(e.state) for e in rows)} carrying a state · "
              f"{len(stmts)} statements ({sum(e.changes is not None for e in stmts)} changing their argument) · "
              f"{sum(e.side == 'agnostic' for e in rows)} model-agnostic, {sum(e.side == 'model' for e in rows)} model-dependent"))


if __name__ == "__main__":
    import sys

    print_catalog(explain="--explain" in sys.argv)
