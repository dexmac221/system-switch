"""Accuracy on the 130 validation questions of Laya v3 (doomLaya training/v3/validation.json).
Run it from the doomLaya directory.
Usage: evaluate.py <predict endpoint> <model> [dump.jsonl]   e.g. evaluate.py http://127.0.0.1:8002/predict llm
With dump.jsonl, one line per question (label, choice, confidence) for deferral_curve.py."""
import collections, json, sys, time, requests
url, model = sys.argv[1], sys.argv[2]
dump = open(sys.argv[3], "w") if len(sys.argv) > 3 else None
import os
rows = json.load(open(os.environ.get("EVAL_DATA", "training/v3/validation.json")))   # EVAL_DATA: e.g. the large test set
ok, tot, conf, lat = collections.Counter(), collections.Counter(), collections.defaultdict(list), []
errors = collections.Counter()
s = requests.Session(); s.trust_env = False
for r in rows:
    t = time.perf_counter()
    o = s.post(url, json={"state": r["state"], "questions": {r["kind"]: r["question"]}, "model": model}, timeout=60).json()
    lat.append((time.perf_counter() - t) * 1000)
    a = o["answers"][r["kind"]]
    right = a["choice"] == r["label"]
    tot[r["kind"]] += 1; ok[r["kind"]] += right
    conf[(r["kind"], right)].append(max(a["probabilities"].values()))
    if dump:
        dump.write(json.dumps({"i": len(lat) - 1, "kind": r["kind"], "label": r["label"], "choice": a["choice"],
                               "conf": max(a["probabilities"].values()), "right": right}) + "\n")
    if not right:
        errors[(r["label"].split("_")[0], a["choice"].split("_")[0])] += 1
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
for k in ("command", "weapon"):
    print(f"{k:8} {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}   median confidence: right {med(conf[(k, True)]):.2f}, wrong {med(conf[(k, False)]):.2f}")
print(f"latency p50 {med(lat):.0f} ms   most frequent errors (label -> choice): {errors.most_common(5)}")


def auroc(pos, neg):
    """P(confidence of a right answer > confidence of a wrong one): 0.5 = confidence says
    nothing about correctness, 1.0 = it separates them perfectly. This is what decides
    whether "stop and think when confidence drops" can work for this actor."""
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


for k in ("command", "weapon"):
    print(f"{k:8} AUROC of confidence (right vs wrong): {auroc(conf[(k, True)], conf[(k, False)]):.3f}")
