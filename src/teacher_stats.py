"""Students trained on a teacher's labels, playing alone (System Switch II, Section 7).

Reads data/teacher/eval_games.jsonl (one line per game: 12 variants x 18 seeds, MAP01, 180 s, the game waits for
the model, so a game is deterministic given checkpoint and seed) and data/teacher/training/<run>.out (the
fine-tune logs: BASE and VALID lines = agreement with the teacher on the held-out seeds 110-111).

Prints: per variant exits with exact (Clopper-Pearson) 95% intervals, mean exit time, deaths, kills, share of
decisions per command type; paired sign tests on the exits (same seeds); the exact permutation test of the two
teachers over runs; validation agreement against exits; where the DAgger runs fall among the ten trained students.

Usage: teacher_stats.py data/teacher
"""
import collections, glob, itertools, json, os, re, sys
from math import comb

D = sys.argv[1]
G = collections.defaultdict(dict)
for l in open(f"{D}/eval_games.jsonl"):
    r = json.loads(l); G[r["variant"]][r["seed"]] = r
ORDER = ["base", "v3", "gemma5", "gemma5b", "gemma5c", "gemma15", "ds5a", "ds5b", "ds5c", "deepseek15", "dagger_a", "dagger_b"]
NAME = {"base": "zero-shot student (Laya typed-decisions)", "v3": "Laya-v3 (trained on the rule oracle)",
        "gemma5": "Gemma teacher, run 1", "gemma5b": "Gemma teacher, run 2", "gemma5c": "Gemma teacher, run 3",
        "gemma15": "Gemma teacher, up to 15 epochs", "ds5a": "DeepSeek teacher, run 1", "ds5b": "DeepSeek teacher, run 2",
        "ds5c": "DeepSeek teacher, run 3", "deepseek15": "DeepSeek teacher, up to 15 epochs",
        "dagger_a": "Gemma + DAgger round, run 1", "dagger_b": "Gemma + DAgger round, run 2"}
TYPES = ["shoot", "collect", "exit", "explore", "open_door", "wait"]


def binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2)
    high = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2)
    return low, high


def sign_test(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def exits(v):
    return sum(g["exit_s"] is not None for g in G[v].values())


print(f"{'variant':42} {'checkpoint':28} {'exits':>6} {'95% exact':>12} {'exit time':>9} {'deaths':>6} {'kills':>5}  decisions by command type")
for v in ORDER:
    g = G[v]; n = len(g); k = exits(v); lo, hi = clopper_pearson(k, n)
    ck = {x["checkpoint"] for x in g.values()}; assert len(ck) == 1 and n == 18 and set(g) == set(range(48, 66))
    assert all(x["status"] == "completed" for x in g.values())
    assert all((x["exit_s"] is not None) == (x["final_map"] == "MAP02") for x in g.values())
    t = [x["exit_s"] for x in g.values() if x["exit_s"] is not None]
    c = collections.Counter()
    for x in g.values():
        c.update(x["commands"])
    tot = sum(c.values())
    print(f"{NAME[v]:42} {ck.pop():28} {k:>2}/{n:<3} [{lo:.2f}, {hi:.2f}] {(sum(t) / len(t) if t else float('nan')):7.0f} s {sum(x['deaths'] for x in g.values()):>6} "
          f"{sum(x['kills'] for x in g.values()):>5}  " + ", ".join(f"{k2} {c[k2] / tot:.0%}" for k2 in TYPES))

print("\ngame time played (a game stops at the exit or after 180 s) and deaths per 10 minutes of play:")
for v in ORDER:
    mins = sum((x["exit_s"] if x["exit_s"] is not None else x["game_seconds"]) for x in G[v].values()) / 60
    d = sum(x["deaths"] for x in G[v].values())
    print(f"   {v:11} {mins:5.1f} min, {d:>2} deaths, {10 * d / mins:4.1f} per 10 min")

print("\npaired on the 18 seeds (exits; sign test on the discordant seeds, two-sided):")
PAIRS = [("base", "ds5b"), ("base", "gemma5"), ("gemma5", "gemma5b"), ("gemma5", "v3"), ("gemma5b", "v3"), ("ds5b", "v3"),
         ("gemma5", "dagger_a"), ("gemma5b", "dagger_a"), ("ds5b", "dagger_a"), ("gemma5", "gemma15"),
         ("gemma5", "deepseek15"), ("gemma15", "deepseek15")]
for a, b in PAIRS:
    x = sum(G[a][s]["exit_s"] is not None and G[b][s]["exit_s"] is None for s in G[a])
    y = sum(G[a][s]["exit_s"] is None and G[b][s]["exit_s"] is not None for s in G[a])
    print(f"   {a:9} {exits(a):>2}/18 vs {b:9} {exits(b):>2}/18: only {a} exits on {x} seeds, only {b} on {y}; p = {sign_test(x, y):.4f}")

single8 = ["gemma5", "gemma5b", "gemma5c", "gemma15", "ds5a", "ds5b", "ds5c", "deepseek15"]
print("\neach single-round student against Laya-v3 on the 18 seeds (sign test):")
for a in single8:
    x = sum(G[a][s]["exit_s"] is not None and G["v3"][s]["exit_s"] is None for s in G[a])
    y = sum(G[a][s]["exit_s"] is None and G["v3"][s]["exit_s"] is not None for s in G[a])
    print(f"   {a:11} {exits(a):>2}/18: only the student on {x} seeds, only Laya-v3 on {y}; p = {sign_test(x, y):.3f}")
tot8 = sum(map(exits, single8))
print(f"   pooled: the eight students exit in {tot8} of {18 * 8} games ({tot8 / 144:.0%}); Laya-v3 in {exits('v3')} of 18")


def chi2_sf(x, k):
    """upper tail of the chi-square distribution with k degrees of freedom (regularized incomplete gamma, series)"""
    from math import exp, lgamma, log
    a, z = k / 2, x / 2
    if z <= 0:
        return 1.0
    term = total = 1 / a; nn = 0
    while abs(term) > 1e-15 * abs(total):
        nn += 1; term *= z / (a + nn); total += term
    return max(0.0, 1 - total * exp(-z + a * log(z) - lgamma(a)))


def cochran_q(variants):
    k = len(variants); seeds = sorted(G[variants[0]])
    X = [[int(G[v][s]["exit_s"] is not None) for v in variants] for s in seeds]
    C = [sum(row[j] for row in X) for j in range(k)]; Rr = [sum(row) for row in X]; N = sum(C)
    den = k * N - sum(r * r for r in Rr)
    Q = (k - 1) * (k * sum(c * c for c in C) - N * N) / den if den else 0.0
    return Q, chi2_sf(Q, k - 1)


print("\nare the students' exit rates on the 18 seeds distinguishable? (Cochran's Q, same seeds)")
for name, vs in (("three Gemma runs, same data and recipe", ["gemma5", "gemma5b", "gemma5c"]),
                 ("three DeepSeek runs, same data and recipe", ["ds5a", "ds5b", "ds5c"]),
                 ("six 5-epoch runs", ["gemma5", "gemma5b", "gemma5c", "ds5a", "ds5b", "ds5c"]),
                 ("all eight single-round students", single8)):
    Q, pv = cochran_q(vs); print(f"   {name:42} Q = {Q:5.2f}, p = {pv:.2f}")
import statistics as _st
ex8 = [exits(v) for v in single8]; rate = sum(ex8) / 144
print(f"   exits of the eight: {ex8}; sample standard deviation {_st.stdev(ex8):.1f}; binomial standard deviation of 18 games at the pooled rate {(18 * rate * (1 - rate)) ** 0.5:.1f}")

gem, ds = ["gemma5", "gemma5b", "gemma5c"], ["ds5a", "ds5b", "ds5c"]
obs = sum(map(exits, gem)) / 3 - sum(map(exits, ds)) / 3
allr = gem + ds; cnt = tot = 0
for comb3 in itertools.combinations(allr, 3):
    rest = [r for r in allr if r not in comb3]
    d = sum(map(exits, comb3)) / 3 - sum(map(exits, rest)) / 3
    tot += 1; cnt += abs(d) >= abs(obs) - 1e-12
print(f"\nteachers, same recipe (5 epochs), three runs each: Gemma {[exits(r) for r in gem]} = {sum(map(exits, gem))}/54, "
      f"deaths {sum(sum(g['deaths'] for g in G[r].values()) for r in gem)}; DeepSeek {[exits(r) for r in ds]} = {sum(map(exits, ds))}/54, "
      f"deaths {sum(sum(g['deaths'] for g in G[r].values()) for r in ds)}")
print(f"   exact permutation test over runs (20 splits), difference of mean exits {obs:+.2f} of 18: two-sided p = {cnt / tot:.2f}")

print("\nvalidation agreement with the teacher (held-out seeds 110-111) against play:")
runs = {}
for f in sorted(glob.glob(f"{D}/training/*.out")):
    name = os.path.basename(f)[:-4]; t = open(f).read()
    base = float(re.search(r"BASE \{'command': ([0-9.]+)\}", t).group(1))
    val = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"'epoch': (\d+), 'validation': \{'command': ([0-9.]+)\}", t)]
    best = max(val, key=lambda x: (x[1], -x[0]))
    runs[name] = (base, best, len(val))
for v in ORDER[2:]:
    base, best, ne = runs[v]
    print(f"   {v:11} before training {base:.3f}; best epoch {best[0]:>2} of {ne:>2}: {best[1]:.3f}  ->  exits {exits(v):>2}/18, "
          f"deaths {sum(g['deaths'] for g in G[v].values())}")


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
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


single = ORDER[2:10]
print(f"   rank correlation of validation agreement with exits over the 8 single-round runs: "
      f"{spearman([runs[v][1][1] for v in single], [exits(v) for v in single]):+.2f}; "
      f"Gemma runs only {spearman([runs[v][1][1] for v in single[:4]], [exits(v) for v in single[:4]]):+.2f}, "
      f"DeepSeek runs only {spearman([runs[v][1][1] for v in single[4:]], [exits(v) for v in single[4:]]):+.2f}")
trained = ORDER[2:]
ex = sorted((exits(v), v) for v in trained)
print(f"\nexits of the ten trained students, sorted: {ex}")
print(f"   the two DAgger runs are the two lowest of ten: probability under exchangeability 1/{comb(10, 2)} = {1 / comb(10, 2):.3f}")
print(f"   exits of the 8 students trained on one round: min {min(exits(v) for v in single)}, max {max(exits(v) for v in single)}; "
      f"deaths min {min(sum(g['deaths'] for g in G[v].values()) for v in single)}, max {max(sum(g['deaths'] for g in G[v].values()) for v in single)}")
