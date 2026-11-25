"""How the students of the second round fail (System Switch II, Section 7.4), from the decisions of the evaluation
games (data/teacher/eval_decisions.jsonl.gz: 10 variants x 18 seeds, MAP01, 180 s, the game waits for the model).

Prints, per variant: the share of explore decisions by game time; when explore is offered, how often it is chosen,
its mean probability, and how often it is chosen with no enemy in sight; the longest run of consecutive explore
decisions per game and in how many games a run of five or more occurs; one example of a long run.

Usage: second_round.py data/teacher
"""
import collections, gzip, json, statistics as st, sys

D = sys.argv[1]
NAME = {"base": "untrained", "v3": "Laya-v3", "gemma5": "Gemma run 1", "gemma5b": "Gemma run 2", "gemma5c": "Gemma run 3",
        "ds5a": "DeepSeek run 1", "ds5b": "DeepSeek run 2", "ds5c": "DeepSeek run 3", "dagger_a": "second round run 1",
        "dagger_b": "second round run 2"}
G = collections.defaultdict(lambda: collections.defaultdict(list))
for r in map(json.loads, gzip.open(f"{D}/eval_decisions.jsonl.gz", "rt")):
    G[r["variant"]][r["seed"]].append(r)
EXITS = collections.Counter()
for l in open(f"{D}/eval_games.jsonl"):
    g = json.loads(l); EXITS[g["variant"]] += g["exit_s"] is not None
WIN = [(0, 30), (30, 60), (60, 120), (120, 181)]
print("share of explore decisions by game time (s), pooled over the 18 games still running in each window")
for v in NAME:
    out = []
    for a, b in WIN:
        d = [r for s in G[v].values() for r in s if a <= r["t"] < b]
        out.append(f"{a}-{min(b, 180)}: {sum(r['choice'] == 'explore' for r in d) / len(d):4.0%}")
    print(f"   {NAME[v]:20} " + " | ".join(out) + f"   exits {EXITS[v]}/18")
print("\nwhen explore is offered (it is always offered together with exit)")
for v in NAME:
    d = [r for s in G[v].values() for r in s if r["p_explore"] is not None]
    c = [r for r in d if r["clear"]]
    assert all("exit" in r["options"] for r in d)
    print(f"   {NAME[v]:20} offered {len(d):5}; chosen {sum(r['choice'] == 'explore' for r in d) / len(d):4.0%}; "
          f"mean P(explore) {st.mean(r['p_explore'] for r in d):.2f}; no enemy in sight: chosen "
          f"{sum(r['choice'] == 'explore' for r in c) / len(c):4.0%} of {len(c)}")
print("\nruns of consecutive explore decisions")
for v in NAME:
    longest, first5, distinct = [], [], []
    for s, L in G[v].items():
        L = sorted(L, key=lambda r: r["t"]); best = cur = 0; f5 = None; span = None
        for i, r in enumerate(L):
            cur = cur + 1 if r["choice"] == "explore" else 0
            if cur > best:
                best, span = cur, (i - cur + 1, i)
            if cur == 5 and f5 is None:
                f5 = r["t"]
        longest.append(best); first5.append(f5)
        if best >= 5:
            run_ = L[span[0]:span[1] + 1]
            distinct.append(len({r["state_id"] for r in run_}) / len(run_))
    f5 = [x for x in first5 if x is not None]
    print(f"   {NAME[v]:20} median longest run {st.median(longest):4.0f}; games with a run of 5 or more {len(f5):2}/18"
          + (f"; median time of the first {st.median(f5):4.0f} s; distinct states in the longest run, median "
             f"{st.median(distinct):.0%}" if f5 else ""))
print("\nexample: the second-round run 1 student on seed 65, its longest run")
L = sorted(G["dagger_a"][65], key=lambda r: r["t"]); best = cur = 0; span = None
for i, r in enumerate(L):
    cur = cur + 1 if r["choice"] == "explore" else 0
    if cur > best:
        best, span = cur, (i - cur + 1, i)
a, b = span
print(f"   {best} explore decisions from {L[a]['t']:.0f} to {L[b]['t']:.0f} s; health {L[a]['hp']:.0f}; no enemy in sight in "
      f"{sum(r['clear'] for r in L[a:b + 1])} of them; options {L[a]['options']}; P(explore) median "
      f"{st.median(r['p_explore'] for r in L[a:b + 1]):.2f}; a health item in sight in {sum(r['health_item_seen'] for r in L[a:b + 1])}")
