"""Exact stage admission and single-process epoch coordination.

    Tool adapters must supply ALL applicable modes, intermediate states and
    resolved-revocation evidence. Missing coverage fails closed. This is not
    distributed fencing, crash recovery, or native Kubernetes enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from threading import Lock
from .authority import AuthoritySpace, Permission, rational
from .contracts import Contract
from .certification import Configuration


@dataclass(frozen=True)
class Request:
    actor: str
    operations: tuple[str, ...]
    targets: frozenset[str]
    epoch: int = 0


@dataclass(frozen=True)
class Stage:
    effective: frozenset[Permission]
    confirmed_absent: frozenset[Permission] = frozenset()
    root_supply: tuple[Q, ...] | None = None


@dataclass(frozen=True)
class Trace:
    contract: Contract
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Evaluation:
    authorized: bool
    complete: bool
    traces: tuple[Trace, ...]


class Guard:
    def __init__(self, space: AuthoritySpace, H, b):
        self.space = space
        self.H = tuple(tuple(map(rational, row)) for row in H)
        self.b = tuple(map(rational, b))
        n = len(space.categories)
        if (len(self.H) != len(self.b) or any(len(row) != n or min(row) < 0 for row in self.H)
                or any(v < 0 for v in self.b)):
            raise ValueError("invalid nonnegative public boundary")
        if not self.safe(space.vector(space.baseline)):
            raise ValueError("baseline violates the public boundary")

    def safe(self, additional):
        if len(additional) != len(self.space.categories) or any(x < 0 for x in additional):
            return False
        baseline = self.space.vector(self.space.baseline, additional=False)
        physical = tuple(s*(r+x) for r, s, x in zip(baseline, self.space.scales, additional))
        return all(sum(h*x for h, x in zip(row, physical)) <= limit
                   for row, limit in zip(self.H, self.b))

    def accepts(self, current, evaluation):
        current = frozenset(current)
        x = self.space.vector(current)
        if not self.space.baseline <= current or not self.safe(x):
            return False
        if not evaluation.authorized or not evaluation.complete or not evaluation.traces:
            return False
        for trace in evaluation.traces:
            if not trace.stages:
                return False
            known, removed = current, frozenset()
            for stage in trace.stages:
                if (not stage.confirmed_absent <= known-self.space.baseline
                        or not known-stage.confirmed_absent <= stage.effective
                        or any(self.space.categories.index(p.category) not in trace.contract.affected
                               for p in stage.effective-known)):
                    return False
                removed |= stage.confirmed_absent & current
                k = self.space.vector(current-removed)
                bound = trace.contract.bound(x, k, stage.root_supply)
                actual = self.space.vector(stage.effective)
                if any(a > f for a, f in zip(actual, bound)) or not self.safe(bound):
                    return False
                known = stage.effective
        return True

    def choose(self, original: Request, candidates, current, evaluate, *, values=None):
        candidates = tuple(candidates)
        if original not in candidates:
            raise ValueError("the original request must be evaluated")
        weights = {t: rational(v) for t, v in (values or dict.fromkeys(original.targets, 1)).items()}
        if set(weights) != original.targets or any(v <= 0 for v in weights.values()):
            raise ValueError("positive, public target values required")
        allowed = []
        for candidate in candidates:
            if (candidate.actor != original.actor or candidate.operations != original.operations
                    or candidate.epoch != original.epoch or not candidate.targets <= original.targets):
                raise ValueError("narrowing cannot change actor, operation, order or epoch")
            if self.accepts(current, evaluate(candidate)):
                cost = sum(weights[t] for t in original.targets-candidate.targets)
                allowed.append((cost, tuple(sorted(candidate.targets)), candidate))
        return min(allowed, key=lambda item: item[:2])[2] if allowed else None


class EpochGate:
    """All local changes must use this lock; external writers are not fenced.

    The operation callback authorizes, admits, reserves and commits while
    holding the lock. New configurations must already have been compiled and
    certified; activation rechecks their exact certificate and switch state.
    """

    def __init__(self, configuration: Configuration, guard: Guard):
        if not configuration.certificate.verifies(configuration.contracts):
            raise ValueError("invalid initial certificate")
        self.configuration, self.guard, self.version = configuration, guard, 0
        self._lock = Lock()

    def call(self, expected_version, operation):
        with self._lock:
            if expected_version != self.version:
                raise RuntimeError("stale epoch")
            return operation(self.configuration)

    def activate(self, expected_version, *, snapshot, compile_configuration):
        with self._lock:
            if expected_version != self.version:
                raise RuntimeError("stale epoch")
            before = snapshot()  # Immutable, complete permissions/provenance/pending state.
            candidate = compile_configuration(before)
            if (not isinstance(before, tuple) or len(before) != 2 or not isinstance(before[1], bytes)
                    or candidate is None or not candidate.certificate.verifies(candidate.contracts)
                    or not self.guard.safe(self.guard.space.vector(before[0]))
                    or snapshot() != before):
                raise ValueError("uncertified configuration or unsafe/drifting switch state")
            self.configuration = candidate
            self.version += 1
            return self.version
