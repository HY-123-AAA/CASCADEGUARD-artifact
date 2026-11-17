"""Five identities in a finite, in-memory authorization model; no LLM or cluster.

    Effects and native authorization are executable set rules. They stand in
    for a trusted tool adapter, not evidence of actual Kubernetes execution.
"""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import combinations
import json

from cascadeguard.authority import AuthoritySpace, Ledger, Permission
from cascadeguard.contracts import Rule, compile_contracts
from cascadeguard.certification import certify, optimize
from cascadeguard.runtime import EpochGate, Evaluation, Guard, Request, Stage, Trace


def model():
    roles = (("O", "coordinate"), ("I", "authorize"), ("D", "deploy"),
             ("N", "connect"), ("R", "recover"))
    permissions = {(agent, domain): Permission(agent, operation, "prod/"+domain, agent+":"+domain)
                   for agent, operation in roles for domain in ("blue", "green")}
    categories = tuple(p.category for p in permissions.values())
    space = AuthoritySpace(categories, (Q(1),)*len(categories))
    rules = []
    for domain in ("blue", "green"):
        for source, target in zip("OIDNR", "IDNRO"):
            rules.append(Rule(source+target+":"+domain, source+"-mode", permissions[source, domain].operation,
                              frozenset({permissions[source, domain]}), frozenset({permissions[target, domain]})))
    # Normal business ends at recovery. The return grant is an optional relation.
    required = frozenset(r.key for r in rules if not r.key.startswith("RO:"))
    return space, permissions, tuple(rules), required


def run_demo():
    space, permissions, rules, required = model()
    q0 = compile_contracts(space, rules)
    configuration = optimize(space, rules, required=required)
    if configuration is None:
        raise RuntimeError("no certified task-preserving configuration")
    selected = tuple(r for r in rules if r.key in configuration.enabled)
    # Public policy: at most one domain of each downstream role is active.
    H = [tuple(Q(c.startswith(actor+":")) for c in space.categories) for actor in "IDNR"]
    guard = Guard(space, H, (1,)*len(H))
    gate = EpochGate(configuration, guard)
    ledger = Ledger(max_ttl=60)
    support = {}
    for domain in ("blue", "green"):
        key = "external:"+domain
        ledger.root(key, permissions["O", domain], now=0, expires=60)
        ledger.confirm(key, "object:"+key, now=0)
        support[permissions["O", domain]] = key
    decisions = []
    for tick, actor in enumerate("OIDNR", start=1):
        original = Request(actor, (permissions[actor, "blue"].operation,), frozenset({"blue", "green"}))
        candidates = [replace(original, targets=frozenset(xs)) for size in (2, 1, 0)
                      for xs in combinations(sorted(original.targets), size)]

        def evaluate(request):
            active = tuple(r for r in selected if r.mode == actor+"-mode"
                           and r.key.split(":")[-1] in request.targets)
            current = ledger.effective()
            authorized = (all(ledger.authorized(permissions[actor, d], now=tick) for d in request.targets)
                          and all(all(ledger.authorized(p, now=tick) for p in r.premises) for r in active))
            outputs = frozenset().union(*(r.outputs for r in active))
            contracts = compile_contracts(space, active)
            if not contracts:  # A no-op has a complete zero-effect contract.
                from cascadeguard.contracts import Contract
                contracts = (Contract(actor+"-mode", ((Q(0),)*len(space.categories),)*len(space.categories), frozenset()),)
            if not configuration.certificate.verifies(contracts):
                return Evaluation(False, False, ())
            return Evaluation(authorized, True, (Trace(contracts[0], (Stage(current | outputs),)),))

        def execute(_configuration):
            request = guard.choose(original, candidates, ledger.effective(), evaluate)
            if request is None:
                raise PermissionError("no safe scope; a new plan must be supplied by the caller")
            for rule in selected:
                if rule.mode == actor+"-mode" and rule.key.split(":")[-1] in request.targets:
                    for permission in rule.outputs:
                        key = rule.key+":"+permission.subject
                        ledger.derive(key, permission, (support[p] for p in rule.premises), now=tick)
                        ledger.confirm(key, "object:"+key, now=tick)
                        support[permission] = key
            return request

        chosen = gate.call(original.epoch, execute)
        decisions.append({"actor": actor, "targets": sorted(chosen.targets), "narrowed": chosen != original})
    recovered = "blue" in chosen.targets and ledger.authorized(permissions["R", "blue"], now=5)
    if not recovered:
        raise AssertionError("original blue-domain recovery task was not preserved")
    return {"q0_certified": certify(q0, len(space.categories)) is not None,
            "qstar_generated": sorted(configuration.enabled), "retention": str(configuration.retained),
            "candidates_checked": configuration.candidates_checked,
            "certificate_exactly_verified": configuration.certificate.verifies(configuration.contracts),
            "decisions": decisions, "task_completed": recovered,
            "effective_permissions": len(ledger.effective())}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
