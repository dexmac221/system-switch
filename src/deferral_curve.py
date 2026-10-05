"""Deferral curve, offline: the actor answers; when its confidence is below a
threshold the decision is deferred to the thinker. For every deferral rate, the
accuracy of the combined system (the plot of ReDAct, arXiv 2604.07036, here on the
Doom validation questions).

Two reference lines: deferring at RANDOM (what confidence adds over chance) and an
ORACLE that defers exactly the actor's mistakes (the ceiling for any trigger).
If the confidence curve is no better than random, "stop and think when unsure"
cannot work for that actor, whatever the thinker.

Usage: deferral_curve.py actor_dump.jsonl thinker_dump.jsonl [--kind command]
"""
import argparse
import json
import random

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("actor")
ap.add_argument("thinker")
ap.add_argument("--kind", default="command")
args = ap.parse_args()

act = {r["i"]: r for r in map(json.loads, open(args.actor)) if r["kind"] == args.kind}
thk = {r["i"]: r for r in map(json.loads, open(args.thinker)) if r["kind"] == args.kind}
ids = sorted(set(act) & set(thk))
n = len(ids)
by_conf = sorted(ids, key=lambda i: act[i]["conf"])          # least confident first
wrong_first = sorted(ids, key=lambda i: act[i]["right"])     # the actor's mistakes first


def acc(deferred):
    d = set(deferred)
    return sum((thk[i] if i in d else act[i])["right"] for i in ids) / n


rng = random.Random(0)
print(f"{args.kind}: n={n}, actor alone {acc([]):.3f}, thinker alone {acc(ids):.3f}")
print(f"{'deferred':>9} {'confidence':>11} {'random':>8} {'oracle':>8}")
for pct in (0, 10, 15, 20, 30, 40, 50, 75, 100):
    k = round(n * pct / 100)
    rnd = sum(acc(rng.sample(ids, k)) for _ in range(200)) / 200
    print(f"{pct:>8}% {acc(by_conf[:k]):>11.3f} {rnd:>8.3f} {acc(wrong_first[:k]):>8.3f}")
