"""All the numbers of the System Switch paper from the per-question dumps of the large
held-out test set (build_testset.py): one place, so the tables cannot drift apart.

  actors   evaluate.py dumps (i, kind, label, choice, conf, right)
  thinkers eval_thinker.py dumps (i, kind, label, choice, right, tokens, ms)

Every interval is a 95% bootstrap that resamples GAMES (`source_run` in the data file),
not questions: states of the same game are correlated.

Usage:
  paper_tables.py --data test_large.json --actor laya=dump_laya.jsonl ... --thinker qwen-think=t.jsonl ...
"""
import argparse
import json
import random
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True)
ap.add_argument("--actor", nargs="+", default=[], help="label=dump.jsonl")
ap.add_argument("--thinker", nargs="+", default=[], help="label=dump.jsonl")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--rates", default="0,10,20,30,50,100")
args = ap.parse_args()

data = json.load(open(args.data))
run_of = [r["source_run"] for r in data]
cat_of = [r["category"] for r in data]


def load(spec):
    label, path = spec.split("=", 1)
    return label, {r["i"]: r for r in map(json.loads, open(path))}


def auroc(rows):
    """Mann-Whitney with ties, O(n log n): P(conf of a right answer > conf of a wrong one)."""
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
        avg = (i + j + 1) / 2                     # 1-based average rank of the tie block
        rank_sum += avg * sum(1 for r in srt[i:j] if r["right"])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def games(ids):
    g = defaultdict(list)
    for i in ids:
        g[run_of[i]].append(i)
    return list(g.values())


def boot(ids, stat):
    """Point value and 95% interval of stat(list of ids), resampling games."""
    units = games(ids)
    rng = random.Random(0)                        # same resamples for every statistic: stable intervals
    vals = []
    for _ in range(args.n):
        sample = [i for u in (rng.choice(units) for _ in units) for i in u]
        v = stat(sample)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:                                  # e.g. AUROC when every answer is right
        return stat(ids), float("nan"), float("nan")
    return stat(ids), vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fmt(t):
    return "--" if t[0] != t[0] else f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"


actors = dict(map(load, args.actor))
thinkers = dict(map(load, args.thinker))

print("== Actors (zero-shot unless trained), large test set")
print(f"{'actor':16} {'n cmd':>5} {'command acc':>21} {'weapon acc':>21} {'AUROC command':>21} {'AUROC weapon':>21}")
for name, d in actors.items():
    cmd = [i for i, r in d.items() if r["kind"] == "command"]
    wpn = [i for i, r in d.items() if r["kind"] == "weapon"]
    acc = lambda ids: sum(d[i]["right"] for i in ids) / len(ids)
    auc = lambda ids: auroc([d[i] for i in ids])
    print(f"{name:16} {len(cmd):>5} {fmt(boot(cmd, acc)):>21} {fmt(boot(wpn, acc)):>21} "
          f"{fmt(boot(cmd, auc)):>21} {fmt(boot(wpn, auc)):>21}")

print("\n== Accuracy by gold category (command: attack/pickup/exit/open_door; weapon: shotgun/pistol)")
cats = sorted({cat_of[i] for d in list(actors.values()) + list(thinkers.values()) for i in d})
print(f"{'model':16} " + " ".join(f"{c:>9}" for c in cats))
for name, d in list(actors.items()) + list(thinkers.items()):
    cells = []
    for c in cats:
        ids = [i for i in d if cat_of[i] == c]
        cells.append(f"{sum(d[i]['right'] for i in ids) / len(ids):9.2f}" if ids else f"{'':>9}")
    print(f"{name:16} " + " ".join(cells))

print("\n== Most frequent command errors (gold -> chosen), share of all command errors")
act = lambda key: (key or "none").split("_")[0]
for name, d in list(actors.items()) + list(thinkers.items()):
    errs = defaultdict(int)
    for i, r in d.items():
        if r["kind"] == "command" and not r["right"]:
            errs[(act(r["label"]), act(r["choice"]))] += 1
    tot = sum(errs.values())
    top = sorted(errs.items(), key=lambda x: -x[1])[:3]
    into_collect = sum(n for (g, c), n in errs.items() if c == "collect") / max(1, tot)
    print(f"{name:16} errors {tot:3}, into collect {into_collect:.0%}: "
          + ", ".join(f"{g}->{c} {n / tot:.0%}" for (g, c), n in top))

print("\n== Thinkers, command questions")
for name, d in thinkers.items():
    cmd = [i for i, r in d.items() if r["kind"] == "command"]
    acc = lambda ids: sum(d[i]["right"] for i in ids) / len(ids)
    tok = sorted(d[i]["tokens"] or 0 for i in cmd)
    ms = sorted(d[i].get("ms") or 0 for i in cmd)
    none = sum(d[i]["choice"] is None for i in cmd)
    print(f"{name:16} n={len(cmd)} acc {fmt(boot(cmd, acc))}  no answer {none}  "
          f"tokens median {tok[len(tok) // 2]}  latency median {ms[len(ms) // 2] / 1000:.1f} s")

rates = [int(x) for x in args.rates.split(",")]
print("\n== Deferral, command questions: the actor's least confident k% go to the thinker")
for tname, t in thinkers.items():
    for aname, a in actors.items():
        ids = sorted(i for i in a if a[i]["kind"] == "command" and i in t)
        if not ids:
            continue

        def combined(sample, pct, how):
            k = round(len(sample) * pct / 100)
            if how == "conf":
                order = sorted(sample, key=lambda i: a[i]["conf"])
            else:                                   # oracle: the actor's mistakes first
                order = sorted(sample, key=lambda i: a[i]["right"])
            d = set(order[:k])
            return sum((t[i] if i in d else a[i])["right"] for i in sample) / len(sample)

        def at_random(sample, pct):
            # exact expectation of deferring k questions chosen uniformly at random
            n = len(sample)
            base = sum(a[i]["right"] for i in sample)
            gain = sum(t[i]["right"] - a[i]["right"] for i in sample)
            return (base + round(n * pct / 100) / n * gain) / n

        print(f"\n{aname} -> {tname}  (n={len(ids)}, actor alone {combined(ids, 0, 'conf'):.3f}, "
              f"thinker alone {combined(ids, 100, 'conf'):.3f})")
        print(f"{'deferred':>9} {'by confidence':>21} {'random':>7} {'oracle':>7} {'conf - random':>21}")
        for pct in rates:
            c = boot(ids, lambda s: combined(s, pct, "conf"))
            diff = boot(ids, lambda s: combined(s, pct, "conf") - at_random(s, pct))
            print(f"{pct:>8}% {fmt(c):>21} {at_random(ids, pct):7.3f} {combined(ids, pct, 'oracle'):7.3f} "
                  f"{fmt(diff):>21}")
