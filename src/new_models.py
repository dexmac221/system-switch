"""Decision models released after the first paper, on its 900 held-out questions (System Switch II, Section 6).

Reads data/new_models/*.jsonl (one line per question, written by eval_decision_models.py in the format of the
first paper's evaluate.py, extras under "x") and the first paper's data (test sets, the records of Qwen3.6 with
reasoning, the records of the two Laya revisions). Same statistics as the first paper: 95% percentile intervals
that resample GAMES (2,000 resamples, seed 0).

With --distinct every distinct question text is counted once (see testset.py).

Usage: new_models.py data/new_models <data directory of github.com/dexmac221/system-switch> [--distinct]
"""
import json, re, sys, random, collections, statistics as st

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testset import keep_indices
DISTINCT = "--distinct" in sys.argv
argv = [x for x in sys.argv[1:] if x != "--distinct"]
R, D = argv[0], argv[1]
test = json.load(open(f"{D}/test_set/test_900.json")); shuf = json.load(open(f"{D}/test_set/test_900_shuffled_options.json"))
runs = [r["source_run"] for r in test]
KEEP = keep_indices(test, DISTINCT)
Q = {r["i"]: r for r in map(json.loads, open(f"{D}/thinkers/qwen36uff_think.jsonl")) if r["kind"] == "command" and r["i"] in KEEP}
print(f"{len(Q)} command and {sum(1 for i in KEEP if test[i]['kind'] == 'weapon')} weapon questions"
      f"{' (distinct texts)' if DISTINCT else ' (as released)'}; gold answer is to collect on "
      f"{sum(test[i]['category'] == 'pickup' for i in Q) / len(Q):.0%} of the command questions")
NEW = [("Laya typed-decisions, classic encoder via laya 0.5", "laya05_classic_typed"), ("Laya base, classic encoder via laya 0.5", "laya05_classic_base"),
       ("Laya 0.5 vega-U, Qwen3.5-0.8B reader", "laya05_vega_08b"), ("Laya 0.5 vega-U, Qwen3.5-4B reader", "laya05_vega_4b"),
       ("pplx-decider-flash v1.3 (4.5B)", "pplx_flash_v13"), ("StartLux v1.1 4B AutoThink, no thought", "startlux4b_never"),
       ("StartLux v1.1 9B AutoThink, no thought", "startlux9b_never")]
ctype = lambda c: "none" if not c else ("open_door" if c.startswith("open") else re.sub(r"_\d+$", "", c))
load = lambda f: {r["i"]: r for r in map(json.loads, open(f"{R}/{f}.jsonl")) if r["i"] in KEEP}
load_v1 = lambda name, order="fixed_order": {r["i"]: r for r in map(json.loads, open(f"{D}/actors/{order}/{name}.jsonl")) if r["i"] in KEEP}


def auroc(rows):
    pos = [r["conf"] for r in rows if r["right"]]; neg = [r["conf"] for r in rows if not r["right"]]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")


def ece(rows, b=15):
    B = collections.defaultdict(list)
    for r in rows: B[min(int(r["conf"] * b), b - 1)].append(r)
    return sum(len(v) / len(rows) * abs(st.mean(r["conf"] for r in v) - st.mean(r["right"] for r in v)) for v in B.values())


def defer(A, ids, k, order=None):
    d = set((order or sorted(ids, key=lambda i: (A[i]["conf"], i)))[:k])
    return st.mean(Q[i]["right"] if i in d else A[i]["right"] for i in ids)


def boot(ids, fn, n=2000):
    g = collections.defaultdict(list)
    for i in ids: g[runs[i]].append(i)
    units = list(g.values()); rng = random.Random(0); v = []
    for _ in range(n):
        v.append(fn([i for u in (rng.choice(units) for _ in units) for i in u]))
    v.sort(); return v[int(0.025 * n)], v[int(0.975 * n) - 1]


def stats_cmd(A, ids):
    """accuracy, AUROC and gain over random deferral at 30% deferred to Qwen3.6, on a sample of questions (repeats allowed)"""
    n = len(ids); k = round(0.3 * n); acc = st.mean(A[i]["right"] for i in ids); thk = st.mean(Q[i]["right"] for i in ids)
    order = sorted(range(n), key=lambda q: (A[ids[q]]["conf"], q))[:k]; d = set(order)
    comb = st.mean(Q[ids[q]]["right"] if q in d else A[ids[q]]["right"] for q in range(n))
    return acc, auroc([A[i] for i in ids]), comb - (acc + k / n * (thk - acc))


print("== New actors on the 600 command questions (doomLaya's option order) and the 300 weapon questions")
print(f"{'model':52} {'command':>18} {'ECE':>5} {'AUROC':>17} {'weapon':>6} {'coll.':>5} {'first':>5} {'30% def.':>8} {'over random':>19}")
for name, f in NEW:
    A = load(f); C = sorted(i for i in A if A[i]["kind"] == "command"); W = [i for i in A if A[i]["kind"] == "weapon"]
    acc, au, gain = stats_cmd(A, C)
    lo_a, hi_a = boot(C, lambda s: st.mean(A[i]["right"] for i in s))
    lo_u, hi_u = boot(C, lambda s: auroc([A[i] for i in s]))
    lo_g, hi_g = boot(C, lambda s: stats_cmd(A, s)[2])
    first = st.mean(A[i]["choice"] == list(test[i]["question"]["criteria"])[0] for i in C)
    coll = st.mean(ctype(A[i]["choice"]) == "collect" for i in C)
    k = round(0.3 * len(C))
    print(f"{name:52} {acc:.3f} [{lo_a:.2f}, {hi_a:.2f}] {ece([A[i] for i in C]):5.2f} {au:.3f} [{lo_u:.2f}, {hi_u:.2f}] "
          f"{st.mean(A[i]['right'] for i in W):6.3f} {coll:5.0%} {first:5.0%} {defer(A, C, k):8.3f} {gain:+.3f} [{lo_g:+.3f}, {hi_g:+.3f}]")

print("\n== Paper I's actors on the same question set (records of the first paper)")
V1 = [("Laya base", "laya-base"), ("Kev 0.8B", "kev-0.8b"), ("Laya typed-decisions", "laya-typed"), ("Kev 4B", "kev-4b"),
      ("Clef-Flash 9B", "clef-flash-9b"), ("Lev 4B", "lev-4b"), ("Kev 9B", "kev-9b"), ("Strands Decider 2B", "strands-2b"),
      ("Julia-1", "julia-1"), ("Laya-v3 (trained)", "laya-v3"), ("doomGemma (trained)", "doomgemma")]
for name, f in V1:
    A = load_v1(f); C = sorted(i for i in A if A[i]["kind"] == "command"); W = [i for i in A if A[i]["kind"] == "weapon"]
    acc, au, gain = stats_cmd(A, C)
    lo_a, hi_a = boot(C, lambda s: st.mean(A[i]["right"] for i in s)); lo_u, hi_u = boot(C, lambda s: auroc([A[i] for i in s]))
    lo_g, hi_g = boot(C, lambda s: stats_cmd(A, s)[2])
    coll = st.mean(ctype(A[i]["choice"]) == "collect" for i in C)
    print(f"{name:52} {acc:.3f} [{lo_a:.2f}, {hi_a:.2f}] {ece([A[i] for i in C]):5.2f} {au:.3f} [{lo_u:.2f}, {hi_u:.2f}] "
          f"{st.mean(A[i]['right'] for i in W):6.3f} {coll:5.0%}       {defer(A, C, round(0.3 * len(C))):8.3f} {gain:+.3f} [{lo_g:+.3f}, {hi_g:+.3f}]")
print("thinkers alone on the command questions: " + ", ".join(
    f"{t} {st.mean(r['right'] for r in map(json.loads, open(f'{D}/thinkers/{t}.jsonl')) if r['kind'] == 'command' and r['i'] in KEEP):.3f}"
    for t in ("qwen36uff_think", "qwen36uff_fast", "gemma4uff_think", "gemma4uff_fast")))

print("\n== What Qwen3.6 with reasoning faces on the questions each new actor defers (30% least confident)")
for name, f in NEW[2:]:
    A = load(f); C = sorted(i for i in A if A[i]["kind"] == "command"); n = len(C); k = round(0.3 * n)
    d = sorted(C, key=lambda i: (A[i]["conf"], i))[:k]
    fix = sum((not A[i]["right"]) and Q[i]["right"] for i in d); brk = sum(A[i]["right"] and not Q[i]["right"] for i in d)
    print(f"   {name:44} thinker on deferred {st.mean(Q[i]['right'] for i in d):.3f} (overall {st.mean(Q[i]['right'] for i in C):.3f}); "
          f"actor on deferred {st.mean(A[i]['right'] for i in d):.3f}; fixes {fix}, breaks {brk}")

print("\n== The classic Laya encoder through the laya 0.5 package against the first paper's records (doomLaya's PyTorch server)")
for f, v1 in (("laya05_classic_typed", "laya-typed"), ("laya05_classic_base", "laya-base")):
    A = load(f); B = load_v1(v1)
    ids = sorted(set(A) & set(B))
    same = st.mean(A[i]["choice"] == B[i]["choice"] for i in ids); dc = [abs(A[i]["conf"] - B[i]["conf"]) for i in ids]
    print(f"   {f:20} vs {v1:10}: same choice on {same:.1%} of the {len(ids)} questions ({sum(A[i]['choice'] != B[i]['choice'] for i in ids)} differ); |confidence difference| median {st.median(dc):.4f}, max {max(dc):.4f}")

print("\n== The built-in gate of Laya 0.5 (vega-U): its abstain flag")
for f in ("laya05_vega_08b", "laya05_vega_4b"):
    A = load(f); C = sorted(i for i in A if A[i]["kind"] == "command")
    ab = [i for i in C if A[i]["x"].get("abstain") in (True, "True")]
    big = [i for i in C if len(A[i]["x"].get("prediction_set") or []) > 1]
    k = len(ab); acc = st.mean(A[i]["right"] for i in C); thk = st.mean(Q[i]["right"] for i in C)
    gate = defer(A, C, k, ab + [i for i in C if i not in set(ab)])
    kept = [i for i in C if i not in set(ab)]
    acc_t = f"{st.mean(A[i]['right'] for i in kept):.3f} on {len(kept)}" if kept else "no question kept"
    print(f"   {f}: abstains on {k} of {len(C)} command questions (prediction set with more than one option: {len(big) / len(C):.0%}); "
          f"accuracy where it does not abstain: {acc_t}")
    W = [i for i in A if A[i]["kind"] == "weapon"]; abw = [i for i in W if A[i]["x"].get("abstain") in (True, "True")]
    tw = [i for i in W if i not in set(abw)]
    print(f"      weapon questions: abstains on {len(abw)} of {len(W)}; accuracy where it abstains "
          f"{(st.mean(A[i]['right'] for i in abw) if abw else float('nan')):.3f}")
    print(f"      deferring the abstentions to Qwen3.6: {gate:.3f} (Qwen3.6 alone: {thk:.3f})")

print("\n== StartLux AutoThink: the same model without and with its thought (600 command questions)")
for sz in ("4b", "9b"):
    N = {i: r for i, r in load(f"startlux{sz}_never").items() if r["kind"] == "command"}; W = load(f"startlux{sz}_always_cmd")
    ids = sorted(set(N) & set(W)); n = len(ids)
    tok = [W[i]["x"]["details"]["thought"]["tokens"] for i in ids]; ms = [W[i]["x"]["details"]["thought"]["ms"] for i in ids]
    a0 = st.mean(N[i]["right"] for i in ids); a1 = st.mean(W[i]["right"] for i in ids)
    lo, hi = boot(ids, lambda s: st.mean(W[i]["right"] for i in s) - st.mean(N[i]["right"] for i in s))
    print(f"   {sz}: no thought {a0:.3f}, always thinking {a1:.3f} (difference {a1 - a0:+.3f} [{lo:+.3f}, {hi:+.3f}]; thought: median {st.median(tok):.0f} tokens, "
          f"{st.median(ms) / 1000:.1f} s); chooses collect {st.mean(ctype(N[i]['choice']) == 'collect' for i in ids):.0%} without and "
          f"{st.mean(ctype(W[i]['choice']) == 'collect' for i in ids):.0%} with the thought")
    for g in (0.6, 0.9):
        d = {i for i in ids if N[i]["x"]["details"]["first_pass"]["top"] < g}
        print(f"      gate {g}: thinks on {len(d) / n:.0%} of the questions, accuracy {st.mean(W[i]['right'] if i in d else N[i]['right'] for i in ids):.3f}")
    def sel_gain(sample, rate):   # thinking on the least confident answers minus thinking on a random share
        m = len(sample); kk = round(rate * m); order = sorted(range(m), key=lambda q: (N[sample[q]]["conf"], q))[:kk]; dd = set(order)
        b0 = st.mean(N[i]["right"] for i in sample); b1 = st.mean(W[i]["right"] for i in sample)
        return st.mean(W[sample[q]]["right"] if q in dd else N[sample[q]]["right"] for q in range(m)) - (b0 + kk / m * (b1 - b0))
    for rate in (0.1, 0.3, 0.5):
        k = round(rate * n); d = set(sorted(ids, key=lambda i: (N[i]["conf"], i))[:k])
        lo, hi = boot(ids, lambda s: sel_gain(s, rate))
        print(f"      thinking on its {rate:.0%} least confident answers: {st.mean(W[i]['right'] if i in d else N[i]['right'] for i in ids):.3f} "
              f"(at random {a0 + k / n * (a1 - a0):.3f}; difference {sel_gain(ids, rate):+.3f} [{lo:+.3f}, {hi:+.3f}])")

print("\n== StartLux with its thought as the thinker behind Paper I's actors (30% least confident deferred)")
for sz in ("4b", "9b"):
  W4 = load(f"startlux{sz}_always_cmd"); print(f"   thinker: StartLux {sz}")
  for v1 in ("laya-base", "laya-typed", "kev-9b", "strands-2b"):
    A = {i: r for i, r in load_v1(v1).items() if r["kind"] == "command"}
    ids = sorted(set(A) & set(W4)); k = round(0.3 * len(ids)); d = set(sorted(ids, key=lambda i: (A[i]["conf"], i))[:k])
    f = lambda T: st.mean(T[i]["right"] if i in d else A[i]["right"] for i in ids)
    print(f"      {v1:11} alone {st.mean(A[i]['right'] for i in ids):.3f}; deferring to StartLux {f(W4):.3f}; to Qwen3.6 {f(Q):.3f}")

print("\n== Option order: StartLux with the options shuffled per question")
for sz in ("4b", "9b"):
    F = load(f"startlux{sz}_never"); S = load(f"startlux{sz}_never_shuffled")
    for name, X, T in (("fixed", F, test), ("shuffled", S, shuf)):
        C = [X[i] for i in X if X[i]["kind"] == "command"]; Wp = [X[i] for i in X if X[i]["kind"] == "weapon"]
        print(f"   {sz} {name:9} command {st.mean(r['right'] for r in C):.3f} AUROC {auroc(C):.3f} collect {st.mean(ctype(r['choice']) == 'collect' for r in C):.0%} "
              f"first option {st.mean(r['choice'] == list(T[r['i']]['question']['criteria'])[0] for r in C):.0%} | weapon {st.mean(r['right'] for r in Wp):.3f}")
