"""Checks asked by a reviewer of the System Switch paper, from the same per-question dumps as
paper_tables.py (command questions only):

  calibration   ECE (15 equal-width bins) and Brier score of the top probability, next to AUROC
  within        AUROC within each gold category (stratified: only right/wrong pairs of the same
                category count), to separate "knows when it is wrong" from "is unsure on hard categories"
  bias          share of errors that choose an item to collect, against the share expected if the
                wrong choice were uniform over the wrong options; share of first-option choices
  selection     choosing the actor (by AUROC) and the deferral rate on half of the games and
                measuring on the other half, over many random splits of the games
  curve         accuracy vs. deferral rate, 0-100% in steps of 5, as CSV for the figure

Usage: review_analyses.py --data test_large.json --actor name=dump ... --thinker name=dump [--curve-csv f]
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True)
ap.add_argument("--actor", nargs="+", required=True, help="label=dump.jsonl (zero-shot first, trained marked with *)")
ap.add_argument("--thinker", required=True, help="label=dump.jsonl")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--splits", type=int, default=500)
ap.add_argument("--curve", nargs="*", default=[], help="actors for the curve CSV")
ap.add_argument("--curve-csv", default=None)
args = ap.parse_args()

data = json.load(open(args.data))
run_of = [r["source_run"] for r in data]
cat_of = [r["category"] for r in data]
keys_of = [list(r["question"]["criteria"]) for r in data]


def load(spec):
    label, path = spec.split("=", 1)
    return label, {r["i"]: r for r in map(json.loads, open(path)) if r["kind"] == "command"}


actors = dict(map(load, args.actor))
tname, thk = load(args.thinker)


def auroc(rows):
    pos = sum(1 for r in rows if r["right"])
    neg = len(rows) - pos
    if not pos or not neg:
        return float("nan")
    srt = sorted(rows, key=lambda r: r["conf"])
    rank_sum, i = 0.0, 0
    while i < len(srt):
        j = i
        while j < len(srt) and srt[j]["conf"] == srt[i]["conf"]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(1 for r in srt[i:j] if r["right"])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def auroc_within(d, ids):
    """Stratified AUROC: concordant right/wrong pairs within the same gold category."""
    num = den = 0.0
    by = defaultdict(list)
    for i in ids:
        by[cat_of[i]].append(d[i])
    for rows in by.values():
        a = auroc(rows)
        if a == a:
            pairs = sum(r["right"] for r in rows) * sum(not r["right"] for r in rows)
            num += a * pairs
            den += pairs
    return num / den if den else float("nan")


def ece(rows, bins=15):
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo < r["conf"] <= hi or (b == 0 and r["conf"] == 0)]
        if sel:
            tot += len(sel) / len(rows) * abs(sum(r["conf"] for r in sel) / len(sel) - sum(r["right"] for r in sel) / len(sel))
    return tot


def games(ids):
    g = defaultdict(list)
    for i in ids:
        g[run_of[i]].append(i)
    return list(g.values())


def boot(ids, stat):
    units, rng, vals = games(ids), random.Random(0), []
    for _ in range(args.n):
        v = stat([i for u in (rng.choice(units) for _ in units) for i in u])
        if v == v:
            vals.append(v)
    vals.sort()
    return stat(ids), vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fmt(t):
    return "--" if t[0] != t[0] else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


print("== Calibration vs. sensitivity (command questions)")
print(f"{'actor':14} {'acc':>5} {'mean conf':>9} {'ECE':>5} {'Brier':>6} {'AUROC':>19} {'AUROC within category':>23}")
for name, d in actors.items():
    ids = sorted(d)
    rows = [d[i] for i in ids]
    acc = sum(r["right"] for r in rows) / len(rows)
    conf = sum(r["conf"] for r in rows) / len(rows)
    brier = sum((r["conf"] - r["right"]) ** 2 for r in rows) / len(rows)
    print(f"{name:14} {acc:5.2f} {conf:9.2f} {ece(rows):5.2f} {brier:6.3f} {fmt(boot(ids, lambda s: auroc([d[i] for i in s]))):>19} "
          f"{fmt(boot(ids, lambda s: auroc_within(d, s))):>23}")

print("\n== Collect bias, normalised by the number of options; position bias")
first_gold = sum(keys_of[i].index(data[i]["label"]) == 0 for i in next(iter(actors.values()))) / len(next(iter(actors.values())))
print(f"gold answer is the first option in {first_gold:.0%} of the command questions (options are not shuffled)")
print(f"{'actor':14} {'errors':>6} {'->collect':>9} {'uniform':>8} {'ratio':>6} {'first option chosen':>20}")
for name, d in actors.items():
    errs = [i for i in d if not d[i]["right"] and d[i]["choice"]]
    obs = sum(d[i]["choice"].startswith("collect") for i in errs) / len(errs)
    exp = sum((sum(k.startswith("collect") for k in keys_of[i]) - data[i]["label"].startswith("collect"))
              / (len(keys_of[i]) - 1) for i in errs) / len(errs)
    first = sum(d[i]["choice"] == keys_of[i][0] for i in d) / len(d)
    print(f"{name:14} {len(errs):6} {obs:9.0%} {exp:8.0%} {obs / exp:6.2f} {first:20.0%}")


def combined(a, ids, k, order_key):
    d = set(sorted(ids, key=order_key)[:k])
    return sum((thk[i] if i in d else a[i])["right"] for i in ids) / len(ids)


def at_random(a, ids, k):
    n = len(ids)
    return (sum(a[i]["right"] for i in ids) + k / n * sum(thk[i]["right"] - a[i]["right"] for i in ids)) / n


print(f"\n== Selection on held-out games ({args.splits} random halves): choose the zero-shot actor by AUROC and the")
print("   deferral rate (10-50%) on one half, measure on the other half, thinker", tname)
zero = {k: v for k, v in actors.items() if not k.endswith("*")}
all_games = sorted(set(run_of[i] for i in next(iter(zero.values()))))
rng = random.Random(1)
chosen, rates_chosen, res = defaultdict(int), defaultdict(int), defaultdict(list)
for _ in range(args.splits):
    A = set(rng.sample(all_games, len(all_games) // 2))
    best = max(zero, key=lambda n: auroc([zero[n][i] for i in zero[n] if run_of[i] in A]))
    a = zero[best]
    ida = [i for i in a if run_of[i] in A and i in thk]
    idb = [i for i in a if run_of[i] not in A and i in thk]
    rate = max((10, 20, 30, 40, 50), key=lambda p: combined(a, ida, round(len(ida) * p / 100), lambda i: a[i]["conf"]))
    chosen[best] += 1
    rates_chosen[rate] += 1
    k = round(len(idb) * rate / 100)
    acc_b = combined(a, idb, k, lambda i: a[i]["conf"])
    res["actor"].append(sum(a[i]["right"] for i in idb) / len(idb))
    res["thinker"].append(sum(thk[i]["right"] for i in idb) / len(idb))
    res["combined"].append(acc_b)
    res["vs random"].append(acc_b - at_random(a, idb, k))
    res["vs best single"].append(acc_b - max(res["actor"][-1], res["thinker"][-1]))
print("actor chosen:", dict(sorted(chosen.items(), key=lambda x: -x[1])), " rate chosen:", dict(sorted(rates_chosen.items())))
for k, v in res.items():
    v = sorted(v)
    print(f"  held-out {k:15} mean {sum(v) / len(v):+.3f}  [{v[int(.025 * len(v))]:+.3f}, {v[int(.975 * len(v)) - 1]:+.3f}]")

if args.curve_csv:
    with open(args.curve_csv, "w") as f:
        f.write("actor,pct,confidence,random,perfect\n")
        for name in args.curve:
            a = actors[name]
            ids = sorted(i for i in a if i in thk)
            for pct in range(0, 101, 5):
                k = round(len(ids) * pct / 100)
                f.write(f"{name},{pct},{combined(a, ids, k, lambda i: a[i]['conf']):.4f},{at_random(a, ids, k):.4f},"
                        f"{combined(a, ids, k, lambda i: a[i]['right']):.4f}\n")
    print("\ncurve ->", args.curve_csv)
