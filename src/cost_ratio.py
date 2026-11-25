"""Cost ratio rho = c_slow / c_fast of the switch, measured end to end in play (added November 2026).

c_fast: the actor's median request latency per game (doomLaya's summary.json); c_slow: the latency of
every thought in the switch logs (games with a language-model thinker; the rule control does not think,
its activations have latency 0 and are left out); the decision period is the median time between requests.

Usage: cost_ratio.py data/games/
"""
import glob, json, os, statistics as st, sys

G = sys.argv[1]
fast, period, slow = {}, {}, {}
for f in glob.glob(f"{G}/runs/*.summary.json"):
    d = json.load(open(f))
    if d.get("latency_ms_median") is not None:
        fast[os.path.basename(f)[:-len(".summary.json")]] = d["latency_ms_median"]
for f in glob.glob(f"{G}/switch_logs/switch_*.jsonl"):
    tag = os.path.basename(f)[len("switch_"):-len(".jsonl")]; ts, th = [], []
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "request":
            ts.append(r["t"])
        elif r.get("kind") == "thought" and r.get("latency_ms"):
            th.append(r["latency_ms"])
    if len(ts) > 1:
        period[tag] = st.median(b - a for a, b in zip(ts, ts[1:]))
    if th:
        slow[tag] = th
tags = sorted(slow)
S = sorted(x for t in tags for x in slow[t])
cf = st.median(fast[t] for t in tags); per = st.median(period[t] for t in tags) * 1000; cs = st.median(S)
q = lambda p: S[min(len(S) - 1, int(p * len(S)))]
print(f"{len(tags)} games with a language-model thinker, {len(S)} thoughts")
print(f"actor: median {cf:.1f} ms per request; decision period {per:.0f} ms")
print(f"thought: median {cs / 1000:.1f} s (10th-90th percentile {q(.1) / 1000:.1f}-{q(.9) / 1000:.1f} s)")
print(f"rho = {cs / cf:.0f} ({q(.1) / cf:.0f}-{q(.9) / cf:.0f}); one thought = {cs / per:.1f} decisions")
