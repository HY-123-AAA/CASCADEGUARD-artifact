"""Per-mode common certificate and finite, implementable configuration search."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations
from scipy.optimize import linprog
from .authority import AuthoritySpace, rational
from .contracts import Contract, Rule, compile_contracts


@dataclass(frozen=True)
class Certificate:
    weights: tuple[Q, ...]
    alpha: Q
    epsilon: Q

    def verifies(self, contracts):
        v, n = self.weights, len(self.weights)
        return (n > 0 and 0 < self.alpha < 1 and self.epsilon > 0
                and sum(v) == 1 and min(v) >= self.epsilon
                and all(len(c.matrix) == n and all(len(row) == n and min(row) >= 0 for row in c.matrix)
                        and all(0 <= i < n and sum(a*w for a, w in zip(c.matrix[i], v)) <= self.alpha*v[i]
                                for i in c.affected) for c in contracts))


def certify(contracts: tuple[Contract, ...], n: int, *, alpha=Q(4, 5), epsilon=Q(1, 1000000)):
    alpha, epsilon = rational(alpha), rational(epsilon)
    if n <= 0 or not 0 < alpha < 1 or not 0 < epsilon <= Q(1, n):
        raise ValueError("invalid strict contraction/positive-weight conditions")
    rows = []
    for c in contracts:
        if (len(c.matrix) != n or any(len(row) != n or min(row) < 0 for row in c.matrix)
                or any(i < 0 or i >= n for i in c.affected)):
            raise ValueError("invalid nonnegative gain matrix")
        for i in c.affected:
            row = list(c.matrix[i])
            row[i] -= alpha
            rows.append(list(map(float, row))+[1.0])
    # Numerical solving proposes a witness; exact rational checking decides.
    # Maximize an interior margin t >= 0 without relaxing any original inequality.
    rhs = [0.0]*len(rows)
    for i in range(n):
        row = [0.0]*(n+1)
        row[i], row[-1] = -1.0, 1.0
        rows.append(row)
        rhs.append(-float(epsilon))
    result = linprog([0.0]*n+[-1.0], A_ub=rows, b_ub=rhs,
                     A_eq=[[1.0]*n+[0.0]], b_eq=[1.0],
                     bounds=[(float(epsilon), None)]*n+[(0.0, None)], method="highs")
    if not result.success:
        return None  # Uncertified, never interpreted as actual unsafety.
    v = tuple(Q(str(float(x))) for x in result.x[:n])
    total = sum(v)
    if total <= 0:
        return None
    certificate = Certificate(tuple(x/total for x in v), alpha, epsilon)
    return certificate if certificate.verifies(contracts) else None


@dataclass(frozen=True)
class Configuration:
    enabled: frozenset[str]
    retained: Q
    contracts: tuple[Contract, ...]
    certificate: Certificate
    candidates_checked: int


def optimize(space: AuthoritySpace, rules: tuple[Rule, ...], *, required=frozenset(),
             values=None, max_optional=16) -> Configuration | None:
    """Enumerate actual rule removals; preserve every declared required rule.

    Removing a rule means disabling its grant relation in this finite backend.
    Required rules are public task minima, not an encoded optimal answer.
    """
    q0 = frozenset(r.key for r in rules)
    if not q0 or len(q0) != len(rules) or not required <= q0:
        raise ValueError("invalid rule IDs/task minimum")
    values = {k: rational(v) for k, v in (values or dict.fromkeys(q0, 1)).items()}
    if set(values) != q0 or any(v <= 0 for v in values.values()):
        raise ValueError("every implementable relation needs a positive public value")
    optional = sorted(q0-required)
    if len(optional) > max_optional:
        raise ValueError("finite demo search budget exceeded")
    best = None
    checked = 0
    for count in range(len(optional)+1):
        for disabled in combinations(optional, count):
            enabled = q0-frozenset(disabled)
            contracts = compile_contracts(space, tuple(r for r in rules if r.key in enabled))
            certificate = certify(contracts, len(space.categories))
            checked += 1
            if certificate is None:
                continue
            retained = sum(values[k] for k in enabled)/sum(values.values())
            if best is None or retained > best.retained:
                best = Configuration(enabled, retained, contracts, certificate, checked)
    if best is None:
        return None
    return Configuration(best.enabled, best.retained, best.contracts, best.certificate, checked)
