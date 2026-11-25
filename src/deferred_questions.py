"""What the thinker faces on the questions an actor defers (analysis added after the paper, November 2026).

For each actor (fixed option order, 600 command questions) the least confident R% of its decisions are
deferred to the thinker. On the deferred questions we report the thinker's accuracy, its difference from
the thinker's accuracy on all questions, the actor errors it fixes, the right answers it turns into
errors, and the net gain over the actor alone. Intervals: 95% percentile bootstrap that resamples GAMES
(`source_run`), 2,000 resamples, seed 0, as in stats_ci.py. Then a summary over all actor x thinker
settings, and rank correlations of the actors' AUROC and calibration (ECE, 15 bins) with the gain over
random deferral and with the thinker's difference.

Usage: deferred_questions.py data/ [--thinker qwen36uff_think] [--rate 0.30]
"""
import argparse, json, os, random
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("data")
ap.add_argument("--thinker", default="qwen36uff_think")
ap.add_argument("--rate", type=float, default=0.30)
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()
D, R = args.data, args.rate
ZERO_SHOT = ["laya-base", "kev-0.8b", "laya-typed", "kev-4b", "clef-flash-9b", "lev-4b", "kev-9b", "strands-2b", "julia-1"]
ACTORS = ZERO_SHOT + ["laya-v3", "doomgemma"]          # laya-gguf is the same model as laya-base
REASONING = ["qwen36uff_think", "gemma4uff_think"]
runs = [r["source_run"] for r in json.load(open(f"{D}/test_set/test_900.json"))]


def load(path):
    return {r["i"]: r for r in map(json.loads, open(path)) if r["kind"] == "command"}


def auroc(rows):
    pos = [r["conf"] for r in rows if r["right"]]; neg = [r["conf"] for r in rows if not r["right"]]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def ece(rows, bins=15):
    b = defaultdict(list)
    for r in rows:
        b[min(int(r["conf"] * bins), bins - 1)].append(r)
    return sum(len(v) / len(rows) * abs(sum(r["conf"] for r in v) / len(v) - sum(r["right"] for r in v) / len(v))
               for v in b.values())


def stats(rows, T):
    n = len(rows); k = round(R * n)
    d = sorted(rows, key=lambda r: (r["conf"], r["i"]))[:k]
    fixes = lambda s: sum((not r["right"]) and T[r["i"]]["right"] for r in s)
    breaks = lambda s: sum(r["right"] and not T[r["i"]]["right"] for r in s)
    th_all = sum(T[r["i"]]["right"] for r in rows) / n
    th_def = sum(T[r["i"]]["right"] for r in d) / k
    net = (fixes(d) - breaks(d)) / n
    random_net = k / n * (fixes(rows) - breaks(rows)) / n      # expected net gain of deferring k at random
    return dict(th_def=th_def, diff=th_def - th_all, fix=fixes(d), brk=breaks(d), net=net, over=net - random_net)


def ci(v):
    v = sorted(v); return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for m in range(i, j + 1):
            r[o[m]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


T = load(f"{D}/thinkers/{args.thinker}.jsonl")
rng = random.Random(0)
print(f"thinker {args.thinker}: {sum(r['right'] for r in T.values()) / len(T):.3f} on all; {R:.0%} deferred; "
      f"intervals resample games ({args.n} resamples)")
print(f"{'actor':14} {'AUROC':>5}  {'thinker on deferred':>20}  {'difference (points)':>22} {'fixed':>5} {'intro':>5}  "
      f"{'net gain (points)':>20}")
per_actor = {}
for a in ACTORS:
    rows = [r for r in load(f"{D}/actors/fixed_order/{a}.jsonl").values() if r["i"] in T]
    groups = defaultdict(list)
    for r in rows:
        groups[runs[r["i"]]].append(r)
    units = list(groups.values()); boot = defaultdict(list)
    for _ in range(args.n):
        s = stats([r for u in (rng.choice(units) for _ in units) for r in u], T)
        for key in ("th_def", "diff", "net"):
            boot[key].append(s[key])
    s = stats(rows, T); per_actor[a] = dict(s, auroc=auroc(rows), ece=ece(rows))
    lo, hi = ci(boot["th_def"]); dlo, dhi = ci(boot["diff"]); nlo, nhi = ci(boot["net"])
    print(f"{a:14} {per_actor[a]['auroc']:5.2f}  {s['th_def']:.2f} [{lo:.2f}, {hi:.2f}]  "
          f"{100 * s['diff']:+6.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}] {s['fix']:5d} {s['brk']:5d}  "
          f"{100 * s['net']:+6.1f} [{100 * nlo:+.1f}, {100 * nhi:+.1f}]")

xs = [per_actor[a]["auroc"] for a in ACTORS]
print(f"\nover the {len(ACTORS)} actors: rank correlation of AUROC with the gain over random "
      f"{spearman(xs, [per_actor[a]['over'] for a in ACTORS]):.2f}, with the thinker's difference "
      f"{spearman(xs, [per_actor[a]['diff'] for a in ACTORS]):.2f}")
z_over = [per_actor[a]["over"] for a in ZERO_SHOT]
print(f"over the {len(ZERO_SHOT)} zero-shot actors: rank correlation of the gain over random with AUROC "
      f"{spearman([per_actor[a]['auroc'] for a in ZERO_SHOT], z_over):.2f}, with calibration (-ECE) "
      f"{spearman([-per_actor[a]['ece'] for a in ZERO_SHOT], z_over):.2f}")

diffs = []
for t in REASONING:
    Tt = load(f"{D}/thinkers/{t}.jsonl")
    for a in ACTORS:
        rows = [r for r in load(f"{D}/actors/fixed_order/{a}.jsonl").values() if r["i"] in Tt]
        diffs.append(stats(rows, Tt)["diff"])
print(f"over the {len(diffs)} settings ({len(ACTORS)} actors x {', '.join(REASONING)}): mean absolute difference "
      f"{100 * sum(map(abs, diffs)) / len(diffs):.1f} points, thinker less accurate on the deferred questions in "
      f"{sum(d < 0 for d in diffs)}")
