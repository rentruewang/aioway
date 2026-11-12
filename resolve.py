"""
resolve.py: lowering a program's intents to concrete methods, by a resolver, and explaining what it chose.

An intent is a config that says what is wanted, not which method (catalog.Intent): it subclasses the group it stands
in for, so it goes wherever a method of that group goes, and it lists the configs it may become.

    loss = Loss()(net(AugmentImage()(x)), y)        # some augmentation of an image: the resolver picks
    loss = Loss()(net(RandomFlip(p=0.5)(x)), y)     # exactly this: never changed

Three parts, each replaceable:

    lower(plan, choice)    one rewrite: each intent replaced by the config chosen for it, wherever it is (a call, an
                           argument, a setting, a loss program). What comes out is an exact program: no intent left.
    Resolver               decides the choice. Default (each intent's first candidate), Given (choices written down:
                           to run the same program again), Exhaustive (every combination, priced), CoordinateDescent
                           (one intent at a time, the others held). The last two ask an oracle what a lowered program
                           costs: loss + price · compute, within a budget.
    Pass                   a physical rewrite of the lowered program: the same result, run differently (KVCache).

    Explain(resolver, passes)(plan)    the program as written, each choice and what it beat, the passes that applied,
                                       and the program that would run: EXPLAIN, without running anything.

The oracle is assumed: a learned cost model, a short proxy run, scaling laws fitted to past runs (ScalingLaw), a
table of measurements, a person.

    python resolve.py      explains an example under each resolver (the oracle's numbers are made up)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Protocol

from torch import Tensor, nn

from catalog import (
    Accuracy, Batches, Beam, KVCache, Choose, ColorJitter, ConvertOp, Fit, GaussianNoise, Greedy, LeftToRight, Intent,
    Layer, LayerOp, LMTree, Loss, Method, MostConfident, Network, Order, PerElement, RandomCrop, RandomFlip, Sample, SSM, Stmt,
    AllAtOnce, Train, Walk, short,
)
from data import Dataset, Grid, Temporal
from program import PROGRAMS, Call, Program, Script, call, calls, inputs, let, update

type Plan = Program | Script
type Choice = Mapping[Intent, Method]

# ═══ Intents ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AugmentImage(Intent, ConvertOp[Grid, Grid]):
    """Some label-preserving random change of an image: a flip, a crop, colour jitter, noise."""
    candidates = (RandomFlip(), RandomCrop(), ColorJitter(), GaussianNoise())


@dataclass(frozen=True)
class Decode(Intent, Choose):
    """Some way to walk an LM's tree: the likeliest token, a sample, a beam."""
    candidates = (Greedy(), Sample(), Beam(k=4))


@dataclass(frozen=True)
class FillOrder(Intent, Order):
    """Some order to fill holes in: all together (diffusion), the surest first, left to right, each its own level."""
    candidates = (AllAtOnce(), MostConfident(), LeftToRight(), PerElement())


@dataclass(frozen=True)
class Mixer(Intent, LayerOp[Temporal, Temporal]):
    """Some way for a sequence's elements to read the earlier ones: attention, a selective scan, a recurrence."""
    candidates = (Layer(nn.MultiheadAttention(64, 4, batch_first=True)), SSM(d=64), Layer(nn.GRU(64, 64, batch_first=True)))


@dataclass(frozen=True)
class TrainLength(Intent, Fit[nn.Module, [Callable[..., Tensor], Dataset[Any] | None]]):
    """Some amount of training: how long to update the network."""
    candidates = (Train(steps=100), Train(steps=200), Train(steps=400))


# ═══ Finding intents, lowering them ═══════════════════════════════════════════════════════════════


def _values(plan: Plan) -> Iterator[Any]:
    """Every constant a program holds: what its calls call, their arguments, the settings of those configs, and
    the same for the programs it uses (its losses, its rounds)."""
    seen: set[int] = set()

    def value(v: Any) -> Iterator[Any]:
        yield v
        if isinstance(v, (Program, Script)):
            yield from program(v)
        elif isinstance(v, Method) and is_dataclass(v):
            for f in fields(v):
                yield from value(getattr(v, f.name))

    def program(p: Plan) -> Iterator[Any]:
        if id(p) in seen:
            return
        seen.add(id(p))
        for e in p.nodes():
            if isinstance(e, Call):
                for v in (e.fn, *e.args, *e.kwargs.values()):
                    yield from value(v)

    yield from program(plan)


def intents(plan: Plan) -> list[Intent]:
    """The intents in a program, each once, in the order they appear. Equal intents are one choice."""
    return list(dict.fromkeys(v for v in _values(plan) if isinstance(v, Intent)))


def _lower(v: Any, choice: Choice) -> Any:
    if isinstance(v, Intent):
        if v not in choice:
            raise KeyError(f"{type(v).__name__}: no choice for it")
        return choice[v]
    if isinstance(v, Method) and is_dataclass(v):
        new = {f.name: _lower(getattr(v, f.name), choice) for f in fields(v)}
        return v if all(new[k] is getattr(v, k) for k in new) else replace(v, **new)
    return v


def lower(plan: Plan, choice: Choice) -> Plan:
    """The program with every intent replaced by the config chosen for it: one rewrite, an exact program out."""

    def change(e: Call) -> Call:
        fn, args, kwargs = _lower(e.fn, choice), tuple(_lower(a, choice) for a in e.args), {k: _lower(v, choice) for k, v in e.kwargs.items()}
        same = fn is e.fn and all(a is b for a, b in zip(args, e.args)) and all(kwargs[k] is v for k, v in e.kwargs.items())
        return e if same else replace(e, fn=fn, args=args, kwargs=kwargs)

    return plan.rewrite(lambda e: isinstance(e, Call), change)


# ═══ Resolvers ════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Cost:
    """What the oracle says a lowered program costs: its expected loss (lower is better) and its compute."""

    loss: float
    compute: float


class Oracle(Protocol):
    def __call__(self, plan: Plan, /) -> Cost: ...


@dataclass(frozen=True)
class Resolution:
    """A resolver's answer: the choice, how it was made, and the choices it priced on the way (none when it priced
    nothing)."""

    choice: dict[Intent, Method]
    how: str
    trials: tuple[tuple[dict[Intent, Method], Cost], ...] = ()
    price: float = 0.0

    def total(self, c: Cost) -> float:
        return c.loss + self.price * c.compute

    def cost(self) -> Cost | None:
        return next((c for ch, c in self.trials if ch == self.choice), None)


class Resolver(Protocol):
    """Decides what each intent in a program becomes."""

    def __call__(self, plan: Plan, /) -> Resolution: ...


@dataclass(frozen=True)
class Default:
    """Each intent its first candidate: no oracle, no search."""

    def __call__(self, plan: Plan, /) -> Resolution:
        return Resolution({i: type(i).candidates[0] for i in intents(plan)}, "default")


@dataclass(frozen=True)
class Given:
    """Choices written down, by intent class: a past resolution, to run the same program again. An intent with no
    choice here falls back to fallback."""

    choices: Mapping[type[Intent], Method]
    fallback: Resolver = Default()

    @staticmethod
    def of(r: Resolution) -> Given:
        return Given({type(i): m for i, m in r.choice.items()})

    def __call__(self, plan: Plan, /) -> Resolution:
        rest = self.fallback(plan).choice
        return Resolution({i: self.choices.get(type(i), rest[i]) for i in intents(plan)}, "given")


@dataclass(frozen=True)
class Exhaustive:
    """Every combination of candidates, lowered and priced; the cheapest within the budget. As many oracle calls as
    the product of the candidates' counts (at most limit)."""

    oracle: Oracle
    price: float = 0.0
    budget: float = math.inf
    limit: int = 1000

    def __call__(self, plan: Plan, /) -> Resolution:
        found = intents(plan)
        combos = list(itertools.islice(itertools.product(*(type(i).candidates for i in found)), self.limit + 1))
        if len(combos) > self.limit:
            raise ValueError(f"more than {self.limit} combinations: use CoordinateDescent")
        trials = tuple((ch, self.oracle(lower(plan, ch))) for ch in (dict(zip(found, combo)) for combo in combos))
        return _cheapest(trials, self.price, self.budget, "exhaustive")


@dataclass(frozen=True)
class CoordinateDescent:
    """From the defaults, one intent at a time: each of its candidates priced with the others held, the cheapest kept;
    rounds until a round changes nothing. Oracle calls: the candidates' sum per round, not their product."""

    oracle: Oracle
    price: float = 0.0
    budget: float = math.inf
    rounds: int = 5

    def __call__(self, plan: Plan, /) -> Resolution:
        choice = Default()(plan).choice
        trials: list[tuple[dict[Intent, Method], Cost]] = []
        for _ in range(self.rounds):
            before = dict(choice)
            for i in choice:
                priced = [({**choice, i: c}, self.oracle(lower(plan, {**choice, i: c}))) for c in type(i).candidates]
                trials += priced
                choice = _cheapest(tuple(priced), self.price, self.budget, "").choice
            if choice == before:
                break
        return replace(_cheapest(tuple(trials), self.price, self.budget, "coordinate descent"), choice=choice)


def _cheapest(trials: tuple[tuple[dict[Intent, Method], Cost], ...], price: float, budget: float, how: str) -> Resolution:
    ok = [(ch, c) for ch, c in trials if c.compute <= budget]
    if not ok:
        raise ValueError("every choice is over the budget")
    best = min(ok, key=lambda t: t[1].loss + price * t[1].compute)[0]
    return Resolution(best, how, trials, price)


# ═══ Physical passes ══════════════════════════════════════════════════════════════════════════════


class Pass(Protocol):
    """A rewrite of a lowered program that keeps its result and changes how it runs."""

    def __call__(self, plan: Plan, /) -> Plan: ...


@dataclass(frozen=True)
class UseKVCache:
    """Each walk of an LM's tree walks it with a KV cache: the same walk, cheaper."""

    def __call__(self, plan: Plan, /) -> Plan:
        def cached(e: Call) -> Call:
            tree = e.args[0]
            return replace(e, args=(call(KVCache(), tree), *e.args[1:]))

        return plan.rewrite(lambda e: isinstance(e, Call) and isinstance(e.fn, Walk) and bool(e.args)
                            and isinstance(e.args[0], Call) and isinstance(e.args[0].fn, LMTree), cached)


# ═══ Explain ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Explanation:
    written: Plan
    resolution: Resolution
    passes: tuple[str, ...]
    runs: Plan

    def __str__(self) -> str:
        r = self.resolution
        lines = [f"EXPLAIN {self.written.name}", "", "as written:", *("  " + l for l in str(self.written).splitlines()), "",
                 f"choices ({r.how}):"]
        for i, m in r.choice.items():
            lines.append(f"  {type(i).__name__}? → {short(m)}")
            priced = {}
            for ch, c in r.trials:                        # each candidate's best total, whatever the others were
                t = r.total(c)
                if t < priced.get(ch[i], math.inf):
                    priced[ch[i]] = t
            for cand, t in sorted(priced.items(), key=lambda kv: kv[1]):
                lines.append(f"      {'✓' if cand == m else ' '} {short(cand):<34} {t:.3f}")
        if (c := r.cost()) is not None:
            lines.append(f"  total {r.total(c):.3f}: loss {c.loss:.3f}, compute {c.compute:g}, {len(r.trials)} programs priced")
        lines += ["", "passes: " + (", ".join(self.passes) or "none applied"), "", "runs:"]
        return "\n".join(lines + ["  " + l for l in str(self.runs).splitlines()])


@dataclass(frozen=True)
class Explain(Stmt):
    """EXPLAIN: what a program would run as, without running it. A statement about a program: it resolves its
    intents, applies the passes, and returns the program as written, the choices with what they beat, the passes
    that changed something, and the exact program that would run."""

    resolver: Resolver = Default()
    passes: tuple[Pass, ...] = (UseKVCache(),)

    def __call__(self, plan: Plan, /) -> Explanation:
        r = self.resolver(plan)
        runs, applied = lower(plan, r.choice), []
        for p in self.passes:
            new = p(runs)
            if new is not runs:
                applied.append(type(p).__name__)
            runs = new
        return Explanation(plan, r, tuple(applied), runs)


# ═══ An example ═══════════════════════════════════════════════════════════════════════════════════


def augmented_classifier() -> Script:
    """An image classifier, its augmentation and its length of training left to a resolver.

        def augmented_classifier(images: Grid, labels: Value) -> Tensor:
            net: Predictor = Network(target=labels)
            TrainLength()(net, augmented_loss, Batches(size=64)(images, labels))   # loss: Loss()(net(AugmentImage()(x)), y)
            return Accuracy()(net(images), labels)
    """
    model, x, y = inputs("network x y")
    augmented_loss = Program("augmented loss", (model, x, y), call(Loss(), model(call(AugmentImage(), x)), y),
                             "the label as the target, each batch augmented first")
    images, labels, net = inputs("images labels network")
    return Script("Augmented classifier", (images, labels), (
        let(net, call(Network, target=labels)),
        update(net, TrainLength(), augmented_loss, call(Batches(size=64), images, labels)),
    ), call(Accuracy(), net(images), labels), "a classifier, its augmentation and its training left to a resolver")


def _oracle(plan: Plan) -> Cost:
    """Made-up numbers, to show the resolvers deciding. The augmentation sets the loss (crop helps most); each doubling
    of the steps takes 0.03 off it, while compute grows with the steps. A real oracle measures, or predicts from what
    it has measured."""
    values = list(_values(plan))
    effect = {RandomFlip: 0.30, RandomCrop: 0.24, ColorJitter: 0.31, GaussianNoise: 0.33}
    loss = next((effect[type(v)] for v in values if type(v) in effect), 0.40)
    steps = [v.steps for v in values if isinstance(v, Train)]
    return Cost(loss - sum(0.03 * math.log2(s / 100) for s in steps), sum(steps) / 100)


if __name__ == "__main__":
    plan = augmented_classifier()
    for resolver in (Default(), Exhaustive(_oracle, price=0.02), CoordinateDescent(_oracle, price=0.02)):
        print(Explain(resolver)(plan), end="\n\n" + "─" * 100 + "\n\n")
    again = Given.of(CoordinateDescent(_oracle, price=0.02)(plan))
    print("Given (the choices written down, to run again):", {k.__name__: short(v) for k, v in again.choices.items()}, end="\n\n")
    gpt = PROGRAMS[0].rewrite(calls(Network), lambda e: e.with_settings(mixer=Mixer()))
    print(Explain(Default())(gpt))
