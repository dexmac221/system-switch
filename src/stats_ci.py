"""Accuracy and AUROC of confidence with bootstrap 95% intervals, from evaluate.py dumps.

With --data (the question file the dump was made on) the bootstrap resamples GAMES
(`source_run`), not questions: states from the same game are correlated, so a
question-level bootstrap would make the intervals too narrow.

Usage: stats_ci.py dump.jsonl [dump2.jsonl ...] [--data test_large.json] [--kind command]
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dumps", nargs="+")
ap.add_argument("--data", default=None)
ap.add_argument("--kind", default="command")
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()

runs = None
if args.data:
    runs = [r["source_run"] for r in json.load(open(args.data))]


def auroc(rows):
    pos = [r["conf"] for r in rows if r["right"]]
    neg = [r["conf"] for r in rows if not r["right"]]
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def ci(values):
    v = sorted(x for x in values if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


rng = random.Random(0)
for path in args.dumps:
    rows = [r for r in map(json.loads, open(path)) if r["kind"] == args.kind]
    if runs:
        groups = defaultdict(list)
        for r in rows:
            groups[runs[r["i"]]].append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in rows]
    acc_b, auc_b = [], []
    for _ in range(args.n):
        sample = [r for u in (rng.choice(units) for _ in units) for r in u]
        acc_b.append(sum(r["right"] for r in sample) / len(sample))
        auc_b.append(auroc(sample))
    acc = sum(r["right"] for r in rows) / len(rows)
    a_lo, a_hi = ci(acc_b)
    u_lo, u_hi = ci(auc_b)
    name = path.rsplit("/", 1)[-1].replace("t900_dump_", "").replace("dump_", "").replace(".jsonl", "")
    print(f"{name:14} n={len(rows):4} units={len(units):3}  acc {acc:.3f} [{a_lo:.3f}, {a_hi:.3f}]   "
          f"AUROC {auroc(rows):.3f} [{u_lo:.3f}, {u_hi:.3f}]")
