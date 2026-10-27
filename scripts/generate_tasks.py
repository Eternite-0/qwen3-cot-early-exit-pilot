#!/usr/bin/env python3
"""Deterministic, verifier-backed synthetic decision tasks for the pilot.

The generator makes task instances first and renders them only after assigning
them to a split. This is target-informed synthetic data, not a benchmark export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

VERSION = "synthetic_decisions_v1.0.0"
SEED = 20261025
SPLIT_SIZES = {"train": 2000, "dev": 300, "pilot_holdout": 500}
FAMILIES = ("rules", "routing", "time_numeric", "multihop", "evidence", "intent")
ANSWER_SPACES = {
    "rules": ["ALLOW", "DENY"],
    "routing": ["R1", "R2", "R3", "NO_ROUTE"],
    "time_numeric": ["WITHIN", "OVER"],
    "multihop": ["REACHABLE", "UNREACHABLE"],
    "evidence": ["SUPPORTED", "INSUFFICIENT"],
    "intent": ["A17", "B42", "C09", "D31"],
}


def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def solve(family: str, p: dict) -> tuple[str, list[str]]:
    """Independent deterministic answer and a compact certificate trace."""
    if family == "rules":
        matched = [r["name"] for r in p["rules"] if all(p["facts"].get(k) == v for k, v in r["if"].items())]
        exception = p["exception"]
        blocked = exception is not None and all(p["facts"].get(k) == v for k, v in exception["if"].items())
        yes = (p["required_rule"] in matched) and not blocked
        return ("ALLOW" if yes else "DENY", [f"matched={','.join(matched) or 'none'}", f"exception_active={str(blocked).lower()}"])
    if family == "routing":
        candidates = [x for x in p["routes"] if all(p["state"].get(k) == v for k, v in x["requires"].items()) and x["capacity"] >= p["load"]]
        candidates.sort(key=lambda x: (x["cost"], x["name"]))
        answer = candidates[0]["name"] if candidates else "NO_ROUTE"
        return answer, [f"feasible={','.join(x['name'] for x in candidates) or 'none'}", f"load={p['load']}"]
    if family == "time_numeric":
        base = p["base_minutes"]
        elapsed = p["elapsed_minutes"]
        limit = p["limit_minutes"]
        value = base + elapsed * p["multiplier"] + p["offset_minutes"]
        answer = "WITHIN" if value <= limit else "OVER"
        return answer, [f"adjusted={value}", f"limit={limit}"]
    if family == "multihop":
        edges = p["edges"]
        reached = {p["source"]}
        frontier = [p["source"]]
        while frontier:
            cur = frontier.pop()
            for a, b in edges:
                if a == cur and b not in reached:
                    reached.add(b)
                    frontier.append(b)
        yes = p["target"] in reached
        return ("REACHABLE" if yes else "UNREACHABLE", [f"reached={','.join(sorted(reached))}", f"target={p['target']}"])
    if family == "evidence":
        present = set(p["evidence"])
        support = set(p["required_evidence"]).issubset(present) and not any(x in present for x in p["contradictory_evidence"])
        return ("SUPPORTED" if support else "INSUFFICIENT", [f"required={','.join(p['required_evidence'])}", f"contradiction={str(bool(present.intersection(p['contradictory_evidence']))).lower()}"])
    if family == "intent":
        # Intent-to-action mapping is shuffled per instance to prevent a fixed label shortcut.
        intent = p["intent"]
        answer = p["mapping"][intent]
        return answer, [f"intent={intent}", f"mapping={intent}:{answer}"]
    raise ValueError(family)


def render(family: str, p: dict, template_id: str) -> tuple[str, list[str]]:
    # Split-exclusive prompt templates. They express the same semantics with
    # different surface structure; template id is provenance, not an input field.
    if template_id.endswith("a"):
        prefix = "Review the case file and apply its stated constraints.\n"
        question = "Select the outcome justified by the facts."
    elif template_id.endswith("b"):
        prefix = "A decision is required from the record below. Use only its rules and facts.\n"
        question = "Which outcome follows?"
    elif template_id.endswith("c"):
        prefix = "Analyze this independent scenario; do not assume facts that are absent.\n"
        question = "Determine the supported result."
    elif template_id.endswith("e"):
        prefix = "Use only the record provided to make this classification.\n"
        question = "Which listed outcome is entailed by the record?"
    elif template_id.endswith("f"):
        prefix = "Review the supplied case facts and apply the listed decision criteria.\n"
        question = "Which single outcome is supported by this record?"
    else:
        prefix = "Process the following operational request using the supplied information.\n"
        question = "What is the correct decision?"

    if family == "rules":
        body = f"Facts: {json.dumps(p['facts'], sort_keys=True)}\nRules: {json.dumps(p['rules'], sort_keys=True)}\nException: {json.dumps(p['exception'], sort_keys=True)}\nRequired rule: {p['required_rule']}"
        labels = ["ALLOW", "DENY"]
    elif family == "routing":
        body = f"State: {json.dumps(p['state'], sort_keys=True)}\nLoad: {p['load']}\nRoutes: {json.dumps(p['routes'], sort_keys=True)}"
        labels = [r["name"] for r in p["routes"]] + ["NO_ROUTE"]
    elif family == "time_numeric":
        body = f"Base duration: {p['base_minutes']} min\nElapsed duration: {p['elapsed_minutes']} min\nMultiplier: {p['multiplier']}\nOffset: {p['offset_minutes']} min\nMaximum allowed: {p['limit_minutes']} min"
        labels = ["WITHIN", "OVER"]
    elif family == "multihop":
        body = f"Dependencies: {json.dumps(p['edges'])}\nStart node: {p['source']}\nTarget node: {p['target']}"
        labels = ["REACHABLE", "UNREACHABLE"]
    elif family == "evidence":
        body = f"Observed evidence: {json.dumps(p['evidence'])}\nRequired evidence: {json.dumps(p['required_evidence'])}\nContradictory evidence: {json.dumps(p['contradictory_evidence'])}"
        labels = ["SUPPORTED", "INSUFFICIENT"]
    elif family == "intent":
        body = f"Request: {p['request']}\nAction mapping: {json.dumps(p['mapping'], sort_keys=True)}"
        labels = list(p["mapping"].values())
    # Candidate order is independently randomized per latent instance, so the
    # answer is not predictable from a stable option position.
    order_seed = int(digest(json.dumps(p, sort_keys=True))[:16], 16)
    random.Random(order_seed).shuffle(labels)
    body = f"Case reference: {p['case_ref']}\n" + body
    return f"{prefix}{body}\n{question}\nAllowed outcomes: {', '.join(labels)}\nReturn only the final outcome label, without explanation.", labels


def instance(family: str, rng: random.Random, index: int) -> dict:
    case_ref = f"F{rng.getrandbits(48):012X}"
    if family == "rules":
        attrs = {f"f{i}": rng.choice(["yes", "no"]) for i in range(4)}
        required = rng.choice(list(attrs))
        secondary = next(k for k in attrs if k != required)
        rules = [{"name": "R1", "if": {required: "yes", secondary: attrs[secondary]}}]
        # Avoid impossible duplicated constraints while retaining a nontrivial exception.
        ex_key = rng.choice([k for k in attrs if k != required])
        exception = {"name": "E1", "if": {ex_key: "yes"}} if rng.random() < 0.5 else None
        return {"facts": attrs, "rules": rules, "exception": exception, "required_rule": "R1", "case_ref": case_ref}
    if family == "routing":
        keys = ["region", "secure"]
        state = {"region": rng.choice(["north", "south"]), "secure": rng.choice(["yes", "no"])}
        load = rng.randint(1, 9)
        routes = []
        for i in range(3):
            routes.append({"name": f"R{i+1}", "requires": {rng.choice(keys): state[rng.choice(keys)]}, "capacity": rng.randint(3, 12), "cost": rng.randint(1, 8)})
        # Normalize accidentally mismatched key/value samples to a valid, generated condition.
        for r in routes:
            k = next(iter(r["requires"]))
            r["requires"][k] = state[k] if rng.random() < 0.55 else ("south" if state[k] == "north" else "north") if k == "region" else ("no" if state[k] == "yes" else "yes")
        return {"state": state, "load": load, "routes": routes, "case_ref": case_ref}
    if family == "time_numeric":
        base = rng.randint(20, 120)
        elapsed = rng.randint(5, 90)
        mult = rng.choice([1, 2, 3])
        offset = rng.randint(-15, 20)
        exact = base + elapsed * mult + offset
        delta = rng.choice([-20, -1, 0, 1, 20])
        return {"base_minutes": base, "elapsed_minutes": elapsed, "multiplier": mult, "offset_minutes": offset, "limit_minutes": max(1, exact + delta), "case_ref": case_ref}
    if family == "multihop":
        nodes = [f"N{i}" for i in range(6)]
        edges = [(nodes[i], nodes[i+1]) for i in range(rng.randint(1, 4))]
        for _ in range(rng.randint(1, 4)):
            a, b = rng.sample(nodes, 2)
            if (a, b) not in edges:
                edges.append((a, b))
        source = nodes[0]
        target = rng.choice(nodes[1:])
        return {"edges": [list(x) for x in edges], "source": source, "target": target, "case_ref": case_ref}
    if family == "evidence":
        pool = [f"E{i}" for i in range(6)]
        req = rng.sample(pool, rng.randint(1, 3))
        contra = [x for x in pool if x not in req][:1]
        evidence = rng.sample(pool, rng.randint(1, 5))
        return {"evidence": evidence, "required_evidence": req, "contradictory_evidence": contra, "case_ref": case_ref}
    if family == "intent":
        intents = ["refund", "reschedule", "cancel", "status"]
        actions = ["A17", "B42", "C09", "D31"]
        rng.shuffle(actions)
        intent = rng.choice(intents)
        utterances = {
            "refund": ["I was charged twice; please return the duplicate payment.", "The item arrived broken and I want my money back.", "Can you reverse the charge for my last order?"],
            "reschedule": ["Move my appointment to a later date.", "I cannot make Tuesday; can we pick another time?", "Please change the delivery slot to Friday."],
            "cancel": ["Stop the pending order before it ships.", "I no longer want this subscription; end it.", "Please withdraw my reservation."],
            "status": ["Has my package shipped yet?", "Where can I see the progress of my request?", "Tell me the current state of the reservation."],
        }
        return {"intent": intent, "request": rng.choice(utterances[intent]), "mapping": dict(zip(intents, actions)), "case_ref": case_ref}
    raise ValueError(family)


def force_answer(family: str, p: dict, answer: str) -> dict:
    """Construct an instance conditioned on a requested label, then verify it."""
    if family == "rules":
        original_exception = p["exception"]
        rule_keys = set(p["rules"][0]["if"])
        free_key = next(k for k in p["facts"] if k not in rule_keys)
        if original_exception is not None:
            # Preserve exception-bearing instances while making it independent
            # of the rule antecedent, so both rule and exception effects vary.
            p["exception"] = {"name":"E1","if":{free_key:original_exception["if"].get(next(iter(original_exception["if"])),"yes")}}
        for key, rule_value in p["rules"][0]["if"].items():
            p["facts"][key] = rule_value if answer == "ALLOW" else ("no" if rule_value == "yes" else "yes")
        if p["exception"] is not None:
            ex_key,ex_value=next(iter(p["exception"]["if"].items()))
            p["facts"][ex_key] = ("no" if ex_value == "yes" else "yes") if answer == "ALLOW" else ex_value
    elif family == "routing":
        state = p["state"]
        if answer == "NO_ROUTE":
            for route in p["routes"]:
                route["requires"] = {"region": "south" if state["region"] == "north" else "north"}
        else:
            for route in p["routes"]:
                route["capacity"] = max(route["capacity"], p["load"] + 1)
                route["requires"] = {"region": "south" if state["region"] == "north" else "north"}
                route["cost"] = 50 + int(route["name"][1:])
            chosen = next(r for r in p["routes"] if r["name"] == answer)
            chosen["requires"] = {"region": state["region"]}
            chosen["cost"] = 1
    elif family == "time_numeric":
        exact = p["base_minutes"] + p["elapsed_minutes"] * p["multiplier"] + p["offset_minutes"]
        p["limit_minutes"] = exact + 1 if answer == "WITHIN" else max(1, exact - 1)
    elif family == "multihop":
        if answer == "REACHABLE":
            p["edges"].append([p["source"], p["target"]])
        else:
            p["edges"] = [e for e in p["edges"] if e[1] != p["target"]]
    elif family == "evidence":
        if answer == "SUPPORTED":
            p["evidence"] = list(p["required_evidence"])
            p["evidence"] = [x for x in p["evidence"] if x not in p["contradictory_evidence"]]
        else:
            missing = next((x for x in p["required_evidence"] if x not in p["contradictory_evidence"]), None)
            if missing is not None:
                p["evidence"] = [x for x in p["evidence"] if x != missing]
            else:
                p["evidence"] = list(set(p["evidence"] + p["contradictory_evidence"]))
    elif family == "intent":
        intent = p["intent"]
        p["mapping"][intent] = answer
    return p


def generate_one(family: str, rng: random.Random, split: str, index: int) -> dict:
    p = instance(family, rng, index)
    answer, trace = solve(family, p)
    template_id = {"train": "template_a", "dev": "template_b", "pilot_holdout": "template_c", "untouched_external": "template_d"}[split]
    prompt, labels = render(family, p, template_id)
    sid = digest(f"{VERSION}|{family}|{json.dumps(p,sort_keys=True)}")[:20]
    return {
        "id": sid, "source_id": sid, "split": split, "family": family,
        "template_family": template_id, "parameters": p, "prompt": prompt,
        "answer": answer, "labels": labels,
        "verifier": "generate_tasks.py::solve/v1", "certificate": trace,
        "difficulty": {
            "family": family,
            "operation_count_proxy": (1 + int(p["exception"] is not None)) if family == "rules" else
                (len(p["routes"])) if family == "routing" else
                (3 + int(p["multiplier"] > 1) + int(abs(p["offset_minutes"]) > 10)) if family == "time_numeric" else
                (len(p["edges"])) if family == "multihop" else
                (len(p["required_evidence"]) + int(bool(set(p["evidence"]) & set(p["contradictory_evidence"])))) if family == "evidence" else 2,
            "input_characters": len(prompt),
            "parameter_seed": rng.getrandbits(32),
        },
        "generator_version": VERSION, "generator_seed": SEED,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def verify(row: dict) -> bool:
    ans, cert = solve(row["family"], row["parameters"])
    p=row["parameters"]; fam=row["family"]
    # Independent verifier implementations intentionally do not call solve().
    if fam == "rules":
        rule_ok=all(p["facts"].get(k)==v for k,v in p["rules"][0]["if"].items())
        exception=p["exception"] is not None and all(p["facts"].get(k)==v for k,v in p["exception"]["if"].items())
        independent="ALLOW" if rule_ok and not exception else "DENY"
    elif fam == "routing":
        eligible=[r for r in p["routes"] if r["capacity"]>=p["load"] and all(p["state"].get(k)==v for k,v in r["requires"].items())]
        independent=min(eligible,key=lambda r:(r["cost"],r["name"]))["name"] if eligible else "NO_ROUTE"
    elif fam == "time_numeric":
        from decimal import Decimal
        value=Decimal(p["base_minutes"])+Decimal(p["elapsed_minutes"])*Decimal(p["multiplier"])+Decimal(p["offset_minutes"])
        independent="WITHIN" if value<=Decimal(p["limit_minutes"]) else "OVER"
    elif fam == "multihop":
        adj={}
        for a,b in p["edges"]: adj.setdefault(a,[]).append(b)
        seen=set(); stack=[p["source"]]
        while stack:
            node=stack.pop()
            if node not in seen: seen.add(node); stack.extend(adj.get(node,[]))
        independent="REACHABLE" if p["target"] in seen else "UNREACHABLE"
    elif fam == "evidence":
        evidence=set(p["evidence"]); required=set(p["required_evidence"]); contradictions=set(p["contradictory_evidence"])
        independent="SUPPORTED" if required-evidence==set() and evidence.isdisjoint(contradictions) else "INSUFFICIENT"
    elif fam == "intent":
        independent=p["mapping"].get(p["intent"])
    else: return False
    return ans == row["answer"] == independent and cert == row["certificate"] and row["id"] == row["source_id"]


def make_external(n: int = 1000, seed: int = 20271025) -> list[dict]:
    """Independent parameter seed and template d; caller must not inspect until frozen."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)]
        p = instance(fam, rng, i)
        target = ANSWER_SPACES[fam][(i // len(FAMILIES)) % len(ANSWER_SPACES[fam])]
        p = force_answer(fam, p, target)
        answer, cert = solve(fam, p)
        if answer != target:
            raise RuntimeError(f"external answer construction failed at {i}")
        prompt, labels = render(fam, p, "template_d")
        sid = digest(f"{VERSION}|untouched|{fam}|{json.dumps(p,sort_keys=True)}")[:20]
        row = {"id": sid, "source_id": sid, "split": "untouched_synthetic_test", "family": fam,
               "template_family": "template_d", "parameters": p, "prompt": prompt, "answer": answer,
               "labels": labels, "certificate": cert, "difficulty": {"family": fam, "input_characters": len(prompt)},
               "generator_version": VERSION, "generator_seed": seed, "verifier": "generate_tasks.py::solve/v1"}
        if not verify(row): raise RuntimeError(f"external verification failed: {sid}")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--external-out", type=Path, default=None)
    ap.add_argument("--external-seed", type=int, default=20271025)
    ap.add_argument("--external-size", type=int, default=1000)
    ap.add_argument("--sizes", nargs=3, type=int, default=list(SPLIT_SIZES.values()), metavar=("TRAIN", "DEV", "PILOT"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    names = ["train", "dev", "pilot_holdout"]
    all_rows = []
    family_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for split, n in zip(names, args.sizes):
        rows = []
        fam_counts = {f: n // len(FAMILIES) + (1 if j < n % len(FAMILIES) else 0) for j, f in enumerate(FAMILIES)}
        fam_i = 0
        for fam in FAMILIES:
            n_fam = fam_counts[fam]
            counts = {label: 0 for label in ANSWER_SPACES[fam]}
            quotas = {label: n_fam // len(counts) + (1 if j < n_fam % len(counts) else 0)
                      for j, label in enumerate(ANSWER_SPACES[fam])}
            accepted = 0
            attempts = 0
            while accepted < n_fam:
                target = next((label for label in ANSWER_SPACES[fam] if counts[label] < quotas[label]), None)
                p = force_answer(fam, instance(fam, rng, fam_i * 10000 + attempts), target)
                ans, _ = solve(fam, p)
                if ans != target:
                    attempts += 1
                    if attempts > n_fam * 500:
                        raise RuntimeError(f"failed to force {fam}:{target}")
                    continue
                row = generate_one(fam, rng, split, fam_i * 10000 + attempts)
                # Rebuild with the forced parameters (generate_one normally draws its own instance).
                template_id = {"train": "template_a", "dev": "template_b", "pilot_holdout": "template_c"}[split]
                prompt, labels = render(fam, p, template_id)
                sid = digest(f"{VERSION}|{fam}|{json.dumps(p,sort_keys=True)}")[:20]
                _, cert = solve(fam, p)
                row.update({"id":sid,"source_id":sid,"parameters":p,"prompt":prompt,"answer":ans,"labels":labels,"certificate":cert,
                            "difficulty":{"family":fam,"operation_count_proxy":len(cert),"input_characters":len(prompt),"parameter_seed":rng.getrandbits(32)}})
                attempts += 1
                ph = digest(row["prompt"])
                if ph in prompt_hashes or row["source_id"] in family_ids or not verify(row):
                    continue
                family_ids.add(row["source_id"])
                prompt_hashes.add(ph)
                counts[row["answer"]] += 1
                rows.append(row)
                accepted += 1
            fam_i += n_fam
        rng.shuffle(rows)
        write_jsonl(args.out / f"{split}.jsonl", rows)
        all_rows.extend(rows)
    summary = {"generator_version": VERSION, "seed": args.seed, "counts": {}, "family_counts": {}, "answer_balance": {}, "cross_split_source_id_overlap": 0}
    for split in names:
        subset = [r for r in all_rows if r["split"] == split]
        summary["counts"][split] = len(subset)
        summary["family_counts"][split] = {f: sum(r["family"] == f for r in subset) for f in FAMILIES}
        summary["answer_balance"][split] = {a: sum(r["answer"] == a for r in subset) for a in sorted({r['answer'] for r in subset})}
    exact = [digest(r["prompt"]) for r in all_rows]
    if len(exact) != len(set(exact)):
        raise RuntimeError("exact prompt duplicate across splits")
    summary["exact_prompt_duplicates"] = 0
    summary["template_policy"] = "split-exclusive template family a/b/c; untouched_external d"
    (args.out / "generation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.external_out:
        external = make_external(args.external_size, args.external_seed)
        write_jsonl(args.external_out, external)
        (args.external_out.parent / "external_generation_summary.json").write_text(json.dumps({"n":len(external),"seed":args.external_seed,"families":{f:sum(x['family']==f for x in external) for f in FAMILIES},"answer_content_not_read_or_evaluated":True},indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
