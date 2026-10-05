"""The thinker as the switch actually uses it: when reasoning ends without an answer (token cap),
switch.py asks again without reasoning. This merges a reasoning dump and a no-reasoning dump of
eval_thinker.py into that behaviour, so the offline numbers match the closed loop.

Usage: thinker_fallback.py think.jsonl fast.jsonl out.jsonl
"""
import json
import sys

think = {r["i"]: r for r in map(json.loads, open(sys.argv[1]))}
fast = {r["i"]: r for r in map(json.loads, open(sys.argv[2]))}
n = used = 0
with open(sys.argv[3], "w") as out:
    for i in sorted(think):
        row = think[i]
        if row["choice"] is None and i in fast:
            row = dict(fast[i], fallback=True, ms=row["ms"] + fast[i]["ms"])
            used += 1
        out.write(json.dumps(row) + "\n")
        n += 1
print(f"{n} rows, {used} answered by the fallback")
