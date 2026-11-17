"""AGCs derived from trusted, finite tool effects, not arbitrary matrix inputs.

The adapter supplies executable AND prerequisites and output sets. This demo
does not infer tool semantics from natural language or discover arbitrary RBAC.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from .authority import AuthoritySpace, Permission, rational


@dataclass(frozen=True)
class Rule:
    key: str
    mode: str
    operation: str
    premises: frozenset[Permission]
    outputs: frozenset[Permission]

    def enabled(self, effective):
        return self.premises <= frozenset(effective)


@dataclass(frozen=True)
class Contract:
    mode: str
    matrix: tuple[tuple[Q, ...], ...]
    affected: frozenset[int]

    def bound(self, current, retained, supply=None):
        n = len(self.matrix)
        current, retained = tuple(map(rational, current)), tuple(map(rational, retained))
        supply = tuple(map(rational, supply)) if supply is not None else (Q(0),)*n
        if (any(len(v) != n for v in (current, retained, supply))
                or any(x < 0 for v in (current, retained, supply) for x in v)
                or any(k > x for k, x in zip(retained, current))
                or any(supply[i] for i in range(n) if i not in self.affected)):
            raise ValueError("invalid current/retained rights or external root supply")
        h = tuple(sum(a*x for a, x in zip(row, current))+u
                  for row, u in zip(self.matrix, supply))
        return tuple(max(retained[i], h[i]) if i in self.affected else current[i]
                     for i in range(n))


def compile_contracts(space: AuthoritySpace, rules: tuple[Rule, ...]) -> tuple[Contract, ...]:
    """Cover unions of all outputs in each mode, including OR overlaps.

    An AND rule is charged to one required, nonbaseline input atom. Summing
    distinct outputs charged to that atom bounds every enabled input subset;
    the category maximum yields kappa, then A_ij=kappa_ij*L_j/L_i.
    Baseline-only effects require a separate supply model and are rejected.
    """
    if len({r.key for r in rules}) != len(rules):
        raise ValueError("duplicate rule ID")
    n = len(space.categories)
    contracts = []
    for mode in sorted({r.mode for r in rules}):
        charged: dict[Permission, set[Permission]] = {}
        affected = set()
        for rule in (r for r in rules if r.mode == mode):
            space.vector(rule.premises | rule.outputs)
            inputs = rule.premises-space.baseline
            if not inputs:
                raise ValueError("baseline-only or unconditional effects need a supply adapter")
            source = min(inputs)
            outputs = rule.outputs-space.baseline
            charged.setdefault(source, set()).update(outputs)
            affected.update(space.categories.index(p.category) for p in outputs)
        matrix = [[Q(0) for _ in range(n)] for _ in range(n)]
        for source, outputs in charged.items():
            j = space.categories.index(source.category)
            for i, category in enumerate(space.categories):
                coefficient = Q(sum(p.category == category for p in outputs))*space.scales[j]/space.scales[i]
                matrix[i][j] = max(matrix[i][j], coefficient)
        contracts.append(Contract(mode, tuple(map(tuple, matrix)), frozenset(affected)))
    return tuple(contracts)
