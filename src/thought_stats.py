"""What the thinker did in closed-loop games, from switch.py logs (switch_<tag>.jsonl).
Per condition: thoughts, how many fell back to no reasoning (reasoning hit the token cap),
thinker latency, the commands it chose, the gate triggers that opened it, and why plans ended.

Usage: thought_stats.py switch_g5-*.jsonl
"""
import json
import re
import sys
from collections import Counter, defaultdict

def label(reason):
    """Gate trigger of a thought: unsure, stuck, blocked, loop, novelty, or a failed plan."""
    if reason.startswith("your plan"):
        return "plan failed"
    if reason.startswith("no progress"):
        return "novelty"
    return reason.split(":")[0]


by_cond = defaultdict(lambda: {"games": 0, "thoughts": 0, "fallback": 0, "lat": [], "choice": Counter(),
                               "reasons": Counter(), "plan_end": Counter()})
for path in sys.argv[1:]:
    tag = re.sub(r"^.*switch_", "", path).replace(".jsonl", "")
    m = re.match(r"(?:[^-]+-)?(.+)-s(\d+)$", tag)
    c = by_cond[m.group(1) if m else tag]
    c["games"] += 1
    for r in map(json.loads, open(path)):
        if r["kind"] == "thought" and r.get("thought"):
            c["thoughts"] += 1
            c["fallback"] += bool(r.get("fallback"))
            c["lat"].append(r["latency_ms"] / 1000)
            c["choice"][(r.get("choice") or "none").split("_")[0]] += 1
            for reason in r.get("reasons") or []:
                c["reasons"][label(reason)] += 1
        elif r["kind"] == "plan_end":
            c["plan_end"][(r.get("why") or "?").split(":")[0]] += 1

for cond, c in by_cond.items():
    lat = sorted(c["lat"])
    p50 = lat[len(lat) // 2] if lat else 0
    print(f"{cond:10} games {c['games']}  thoughts {c['thoughts']}  fallback {c['fallback']}  "
          f"latency p50 {p50:.1f} s  max {max(lat) if lat else 0:.1f} s")
    print(f"           chose {dict(c['choice'].most_common())}")
    print(f"           triggers {dict(c['reasons'].most_common())}")
    if c["plan_end"]:
        print(f"           plans ended by {dict(c['plan_end'].most_common())}")
