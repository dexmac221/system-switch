"""System Switch II -- one harness for the decision models released after the first paper.

Reads the 900 held-out questions of the first paper (data/test_set/test_900.json of github.com/dexmac221/
system-switch) and writes one JSON line per question in the format of src/evaluate.py (i, kind, label, choice,
conf = top probability, right), so that paper_tables.py, deferral_curve.py and deferred_questions.py read the
records unchanged. Extra fields go under "x": abstain / prediction set / route / unbound (vegaml), whether a
thought was used, its tokens and the latency (StartLux AutoThink), and the raw confidence the model reports.

Backends (the question goes in unchanged: the state as text, the question with doomLaya's instructions+criteria):
  http  any /v1/systemone server, e.g. startlux-decide on llama-server; --extra adds request fields
        (StartLux: {"think_mode": "never"|"always"|"auto", "gate": 0.6, "think_budget": 4096})
  vega  vegaml.load(<name>): Laya 0.5's default engine (vega-U over a frozen Qwen3.5 reader)
  laya  laya.load(<dir>, subfolder=...): the classic Laya encoder through the 0.5 package
  pplx  pplx_decider.Decider.from_pretrained(<dir>): Perplexity's decider (protocol v4)

Usage: eval_decision_models.py --data test_900.json --out records.jsonl --backend http --url http://127.0.0.1:8090/v1/systemone
       [--extra '{"think_mode": "never"}'] [--kind command|weapon|all] [--workers 4] [--limit N]
The output is appended and resumable: questions already in --out are skipped.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--backend", required=True, choices=["http", "vega", "laya", "pplx"])
ap.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
ap.add_argument("--extra", default="{}", help="JSON merged into every http request")
ap.add_argument("--name", default=None, help="vega: vegaml.load() argument; laya/pplx: model directory")
ap.add_argument("--subfolder", default=None, help="laya: typed-decisions, or none for the English base")
ap.add_argument("--kind", default="all", choices=["command", "weapon", "all"])
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--timeout", type=float, default=900)
args = ap.parse_args()

rows = json.load(open(args.data))
todo = [(i, r) for i, r in enumerate(rows) if args.kind == "all" or r["kind"] == args.kind]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["i"] for l in open(args.out) if l.strip()}
todo = [(i, r) for i, r in todo if i not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(todo)} questions to answer ({len(done)} already in {args.out})", file=sys.stderr)


def probs_of(a):
    return a.get("probabilities") or a.get("probs") or {}


if args.backend == "http":
    import requests
    extra = json.loads(args.extra)
    local = threading.local()

    def answer(state, questions):
        if not hasattr(local, "s"):
            local.s = requests.Session(); local.s.trust_env = False
        t0 = time.perf_counter()
        r = local.s.post(args.url, json={"state": state, "questions": questions, **extra}, timeout=args.timeout)
        r.raise_for_status()
        out = r.json()
        out["_ms"] = (time.perf_counter() - t0) * 1000
        return out
elif args.backend == "vega":
    import vegaml
    model = vegaml.load(args.name) if args.name else vegaml.load()
    lock = threading.Lock()

    def answer(state, questions):
        with lock:
            t0 = time.perf_counter()
            out = model.decide(state, questions)
            out["_ms"] = (time.perf_counter() - t0) * 1000
            return out
elif args.backend == "laya":
    import laya
    model = laya.load(args.name, subfolder=args.subfolder) if args.subfolder else laya.load(args.name)
    lock = threading.Lock()

    def answer(state, questions):
        with lock:
            t0 = time.perf_counter()
            out = model.predict(state, questions)
            out["_ms"] = (time.perf_counter() - t0) * 1000
            return out
else:
    sys.path.insert(0, os.path.join(args.name, "source", "src"))
    from pplx_decider import Decider
    model = Decider.from_pretrained(args.name, device="cuda")
    lock = threading.Lock()

    def answer(state, questions):
        with lock:
            t0 = time.perf_counter()
            out = model.decide({"state": state, "questions": questions})
            out["_ms"] = (time.perf_counter() - t0) * 1000
            return out


EXTRA_KEYS = ("confidence", "abstain", "prediction_set", "route", "unbound", "readout", "source", "thought",
              "thought_tokens", "thought_used", "thinking", "details", "first_readout")


def one(i, r):
    out = answer(r["state"], {r["kind"]: r["question"]})
    a = out["answers"][r["kind"]]
    p = probs_of(a)
    choice = a.get("choice")
    rec = {"i": i, "kind": r["kind"], "label": r["label"], "choice": choice,
           "conf": max(p.values()) if p else None, "right": choice == r["label"]}
    x = {k: a[k] for k in EXTRA_KEYS if k in a}
    for k in ("usage", "details", "thought", "model"):
        if k in out and k not in x:
            x[k] = out[k]
    x["ms"] = round(out["_ms"], 1)
    x["probs"] = p
    rec["x"] = x
    return rec


wlock = threading.Lock()
n = ok = 0
t0 = time.time()
with open(args.out, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(one, i, r): i for i, r in todo}
    for fut in as_completed(futs):
        try:
            rec = fut.result()
        except Exception as e:  # recorded, not fatal: a failed question counts as unanswered
            print(f"question {futs[fut]}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            continue
        with wlock:
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1; ok += rec["right"]
            if n % 50 == 0:
                print(f"{n}/{len(todo)}  accuracy so far {ok / n:.3f}  {time.time() - t0:.0f} s", file=sys.stderr)
print(f"done: {n} answered, accuracy {ok / max(1, n):.3f}, {time.time() - t0:.0f} s", file=sys.stderr)
