"""A question evaluator as the deferral gate (System Switch II, Section 5).

The gate of the first paper defers the questions on which the actor is least confident. Here the questions deferred
are those on which the thinker is predicted to FIX the actor. For every (actor, question): +1 if the actor is wrong
and the thinker right (a fix), -1 if the actor is right and the thinker wrong (a break), 0 otherwise. Two logistic
regressions (P fix, P break) on features available AT DECISION TIME (the state, the options offered, the actor's
output; never the label, the gold category or the thinker's answer); score = P(fix) - P(break).

Cross-validation by GAMES (7 folds of 6 games, 10 repetitions): every question is scored by a model that has not
seen its game. Compared per actor at the same deferral rate: the actor's confidence (first paper), a hand rule
(defer the actor's "collect" choices first, least confident first), the evaluator with three feature sets, an
evaluator that has never seen the actor it gates (leave one actor out, no actor identity, still scored on held-out
games), random deferral (expected value) and the value oracle (fixes first, breaks last).
Then: other rates; 95% intervals of the differences that resample games (each statistic is the mean over the 10
repetitions); the largest coefficients.

Identical question texts are always kept in the same fold. With --distinct every distinct text is counted once.

Usage: question_evaluator.py <data directory of github.com/dexmac221/system-switch> [thinker=qwen36uff_think] [--distinct]
"""
import json, random, re, sys
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testset import first_occurrence, keep_indices
DISTINCT = "--distinct" in sys.argv
argv = [x for x in sys.argv[1:] if x != "--distinct"]
D = argv[0]; TH = argv[1] if len(argv) > 1 else "qwen36uff_think"
ACT = ["laya-base", "laya-v3", "doomgemma", "kev-0.8b", "laya-typed", "kev-4b", "clef-flash-9b", "lev-4b",
       "kev-9b", "strands-2b", "julia-1"]
TYPES = ["collect", "shoot", "open_door", "exit", "explore", "retreat", "wait"]
test = json.load(open(f"{D}/test_set/test_900.json"))
KEEP = keep_indices(test, DISTINCT); FIRST = first_occurrence(test)
T = {r["i"]: r for r in map(json.loads, open(f"{D}/thinkers/{TH}.jsonl")) if r["kind"] == "command" and r["i"] in KEEP}
ids = sorted(T); n = len(ids)
runs = sorted({test[i]["source_run"] for i in range(len(test)) if test[i]["kind"] == "command"})
run_of = {i: runs.index(test[i]["source_run"]) for i in ids}
# identical texts are kept in one fold: a question is assigned to the game of the first question with its text
fold_game = {i: runs.index(test[FIRST[i]]["source_run"]) for i in ids}


def ctype(c):
    if not c:
        return "none"
    return "open_door" if c.startswith("open") else re.sub(r"_\d+$", "", c)


def num(pat, s, d=0.0):
    m = re.search(pat, s); return float(m.group(1)) if m else d


def question_features(i):
    s = test[i]["state"]; keys = list(test[i]["question"]["criteria"])
    dist = [float(x) for x in re.findall(r"at ([0-9.]+) meters", s)]
    f = {"hp": num(r"has (\d+) health", s), "armor": num(r"and (\d+) armor", s),
         "pistol_ammo": num(r"pistol with (\d+) ammo", s), "shotgun_ammo": num(r"shotgun with (\d+) ammo", s),
         "no_enemy": float("clear of enemies" in s), "n_items": float(len(dist)),
         "nearest": min(dist) if dist else 30.0, "unreachable": float("Unreachable" in s),
         "prev_blocked": float(bool(re.search(r"Previous command result: (No navigable|.*block)", s))),
         "door_in_state": float("door" in s.lower()), "key_in_state": float("key" in s.lower()),
         "state_len": len(s) / 100, "n_opt": float(len(keys))}
    for t in TYPES:
        f["opt_" + t] = float(sum(ctype(k) == t for k in keys))
    return f


def actor_features(a, r, i):
    keys = list(test[i]["question"]["criteria"])
    f = {"conf": r["conf"], "conf_x_nopt": r["conf"] * len(keys), "first": float(r["choice"] == keys[0])}
    for t in TYPES:
        f["chose_" + t] = float(ctype(r["choice"]) == t)
    for b in ACT:
        f["actor_" + b] = float(a == b)
    return f


A = {a: {r["i"]: r for r in map(json.loads, open(f"{D}/actors/fixed_order/{a}.jsonl")) if r["kind"] == "command"} for a in ACT}
rows = []  # (actor, question, game, features, value)
for a in ACT:
    for i in ids:
        r = A[a][i]; t = T[i]["right"]
        v = 1 if (not r["right"] and t) else (-1 if (r["right"] and not t) else 0)
        rows.append((a, i, run_of[i], {**question_features(i), **actor_features(a, r, i)}, v))
names = list(rows[0][3])
SETS = {"question": lambda k: not k.startswith(("conf", "first", "chose_")),      # state + options + which actor asks
        "actor": lambda k: k.startswith(("conf", "first", "chose_", "actor_")),   # the actor's output only
        "all": lambda k: True}
X_all = np.array([[r[3][k] for k in names] for r in rows]); V = np.array([r[4] for r in rows])
GAME = np.array([fold_game[r[1]] for r in rows]); ACTOR = np.array([r[0] for r in rows])
by_actor = {a: np.where(ACTOR == a)[0] for a in ACT}


def logit_fit(X, y, l2=1.0, it=400, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y)); b -= lr * (p - y).mean()
    return w, b


def fit_score(X, tr, te):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    wf, bf = logit_fit(Z[tr], (V[tr] == 1).astype(float)); wb, bb = logit_fit(Z[tr], (V[tr] == -1).astype(float))
    return 1 / (1 + np.exp(-(Z[te] @ wf + bf))) - 1 / (1 + np.exp(-(Z[te] @ wb + bb)))


def folds(seed):
    rng = random.Random(seed); order = list(range(len(runs))); rng.shuffle(order)
    return {g: k % 7 for k, g in enumerate(order)}


def cv_scores(cols, seed, leave_actor_out=False):
    X = X_all[:, [names.index(c) for c in cols]]; fold = folds(seed); sc = np.zeros(len(rows))
    fg = np.array([fold[g] for g in GAME])
    for k in range(7):
        if not leave_actor_out:
            sc[fg == k] = fit_score(X, fg != k, fg == k)
        else:
            for a in ACT:
                te = (fg == k) & (ACTOR == a)
                sc[te] = fit_score(X, (fg != k) & (ACTOR != a), te)
    return sc


def accuracy(a, order, k):
    d = set(order[:k])
    return np.mean([T[i]["right"] if i in d else A[a][i]["right"] for i in ids])


def order_by_score(a, sc):
    return [rows[j][1] for j in sorted(by_actor[a], key=lambda j: -sc[j])]


def order_by_conf(a):
    return sorted(ids, key=lambda i: (A[a][i]["conf"], i))


def order_collect_rule(a):
    return sorted(ids, key=lambda i: (ctype(A[a][i]["choice"]) != "collect", A[a][i]["conf"], i))


def order_oracle(a):
    val = {rows[j][1]: rows[j][4] for j in by_actor[a]}
    return sorted(ids, key=lambda i: -val[i])


R = 0.30; k = round(R * n); thinker_acc = np.mean([T[i]["right"] for i in ids])
REPS = 10
print(f"thinker {TH} ({thinker_acc:.3f} alone), {n} command questions{' (distinct texts)' if DISTINCT else ' (as released)'}, "
      f"{len(set(run_of.values()))} games, {len(ACT)} actors; "
      f"{R:.0%} deferred; cross-validation by games (7 folds), mean of {REPS} repetitions")
SC = {s: [cv_scores([c for c in names if flt(c)], rep) for rep in range(REPS)] for s, flt in SETS.items()}
no_id = [c for c in names if not c.startswith("actor_")]
SC["unseen actor"] = [cv_scores(no_id, rep, leave_actor_out=True) for rep in range(REPS)]
res = {s: {a: [accuracy(a, order_by_score(a, sc), k) for sc in SC[s]] for a in ACT} for s in SC}
print(f"{'actor':14} {'alone':>6} {'random':>6} {'confid.':>7} {'collect':>7} {'e.quest':>7} {'e.actor':>7} {'e.all':>6} {'e.unseen':>8} {'oracle':>6}")
tot = {x: [] for x in ("random", "confid.", "collect", "question", "actor", "all", "unseen actor", "oracle")}
for a in ACT:
    alone = np.mean([A[a][i]["right"] for i in ids])
    vals = {"random": alone + k / n * (thinker_acc - alone), "confid.": accuracy(a, order_by_conf(a), k),
            "collect": accuracy(a, order_collect_rule(a), k), "oracle": accuracy(a, order_oracle(a), k),
            **{s: np.mean(res[s][a]) for s in SC}}
    for x in tot:
        tot[x].append(vals[x])
    print(f"{a:14} {alone:6.3f} {vals['random']:6.3f} {vals['confid.']:7.3f} {vals['collect']:7.3f} {vals['question']:7.3f} "
          f"{vals['actor']:7.3f} {vals['all']:6.3f} {vals['unseen actor']:8.3f} {vals['oracle']:6.3f}")
print(f"{'mean':14} {'':6} {np.mean(tot['random']):6.3f} {np.mean(tot['confid.']):7.3f} {np.mean(tot['collect']):7.3f} "
      f"{np.mean(tot['question']):7.3f} {np.mean(tot['actor']):7.3f} {np.mean(tot['all']):6.3f} {np.mean(tot['unseen actor']):8.3f} "
      f"{np.mean(tot['oracle']):6.3f}")
print("spread across the repetitions (max - min, points, mean over actors): " +
      ", ".join(f"{s} {100 * np.mean([max(res[s][a]) - min(res[s][a]) for a in ACT]):.1f}" for s in SC))

for Rx in (0.10, 0.30, 0.50):
    kx = round(Rx * n)
    dv = [np.mean([accuracy(a, order_by_score(a, sc), kx) for sc in SC["all"]]) - accuracy(a, order_by_conf(a), kx) for a in ACT]
    print(f"{Rx:.0%} deferred: evaluator (all) - confidence, mean over actors {100 * np.mean(dv):+.1f} points "
          f"(better for {sum(d > 0 for d in dv)} of {len(dv)} actors)")

# ---- intervals: resample games; every statistic is the mean over the repetitions of the cross-validation ----
pos = {i: q for q, i in enumerate(ids)}
game_idx = {}
for i in ids:
    game_idx.setdefault(run_of[i], []).append(pos[i])
game_idx = [np.array(v) for v in game_idx.values()]
TR = np.array([T[i]["right"] for i in ids], dtype=float)
AR = {a: np.array([A[a][i]["right"] for i in ids], dtype=float) for a in ACT}
# sort keys as arrays over question positions: lower key = deferred first
KEY = {"confidence": {a: np.array([A[a][i]["conf"] for i in ids]) + 1e-9 * np.arange(n) for a in ACT},
       "collect rule": {a: np.array([(ctype(A[a][i]["choice"]) != "collect") * 10 + A[a][i]["conf"] for i in ids]) + 1e-9 * np.arange(n) for a in ACT}}
for s in SC:
    KEY[s] = {a: [-np.array([sc[j] for j in by_actor[a]]) for sc in SC[s]] for a in ACT}   # rows of an actor are in ids order


def acc_from_key(a, key, samp):
    kb = round(R * len(samp)); sel = np.argpartition(key[samp], kb - 1)[:kb]
    right = AR[a][samp].copy(); right[sel] = TR[samp][sel]
    return right.mean()


def method_acc(a, m, samp):
    ks = KEY[m][a]
    return np.mean([acc_from_key(a, kk, samp) for kk in ks]) if isinstance(ks, list) else acc_from_key(a, ks, samp)


def interval(m, base):
    rng = np.random.default_rng(0); full = np.arange(n)
    point = {a: method_acc(a, m, full) - method_acc(a, base, full) for a in ACT}
    boot = {a: [] for a in ACT}; mean = []
    for _ in range(2000):
        samp = np.concatenate([game_idx[g] for g in rng.integers(0, len(game_idx), len(game_idx))])
        d = [method_acc(a, m, samp) - method_acc(a, base, samp) for a in ACT]
        for a, x in zip(ACT, d):
            boot[a].append(x)
        mean.append(np.mean(d))
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    print(f"evaluator ({m}) - {base} at {R:.0%}: point estimate and 95% interval (2,000 resamples of games)")
    for a in ACT:
        lo, hi = ci(boot[a]); print(f"   {a:14} {100 * point[a]:+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    lo, hi = ci(mean); print(f"   {'mean':14} {100 * np.mean(list(point.values())):+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")


interval("all", "confidence")
interval("all", "collect rule")
interval("collect rule", "confidence")
interval("unseen actor", "confidence")

Z = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-9)
wf, _ = logit_fit(Z, (V == 1).astype(float)); wb, _ = logit_fit(Z, (V == -1).astype(float)); w = wf - wb
print("largest standardized coefficients, fix minus break (+ pushes to defer, - to keep the actor's answer):")
print("   " + ", ".join(f"{names[j]} {w[j]:+.2f}" for j in np.argsort(-np.abs(w))[:12]))
print(f"labels over all actor-question pairs: fixes {np.mean(V == 1):.1%}, breaks {np.mean(V == -1):.1%}")
