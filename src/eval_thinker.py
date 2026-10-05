"""The thinker as a decision maker, zero-shot: the same 130 validation questions as
evaluate.py, answered by a chat LLM (OpenAI-compatible, e.g. llama-server) that may
reason before answering. The answer is constrained to the option keys (json schema).

The question this answers: when the actor defers, is the thinker actually right?
"A model that reasons could correct the actor without training" is a hypothesis;
this is the measurement. Run it with and without reasoning (--think).

Usage (from the doomLaya directory):
  eval_thinker.py --llm http://127.0.0.1:8090 --label qwen38-think --think --dump thinker.jsonl
With --workers N the questions are sent N at a time (the server needs -np N);
--kind command keeps only the command questions. `i` is always the index in the
full question file, so the dump joins with evaluate.py dumps.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--llm", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--think", action="store_true", help="let the model reason before answering")
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--dump", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--kind", default=None, help="only this question kind (command or weapon)")
ap.add_argument("--resume", action="store_true", help="keep the rows already in --dump and ask only the others")
args = ap.parse_args()

rows = list(enumerate(json.load(open(os.environ.get("EVAL_DATA", "training/v3/validation.json")))))
if args.kind:
    rows = [(i, r) for i, r in rows if r["kind"] == args.kind]
if args.limit:
    rows = rows[:args.limit]


def ask(item):
    i, r = item
    s = requests.Session()
    s.trust_env = False
    q = r["question"]
    keys = list(q["criteria"])
    prompt = (f"You control a DOOM player.\n\nSituation:\n{r['state']}\n\nQuestion: {q.get('instructions', '')}\n\n"
              "Options:\n" + "\n".join(f"- {k}: {q['criteria'][k]}" for k in keys)
              + "\n\nAnswer with the key of the best option.")
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": keys}}, "required": ["choice"]}
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": args.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": args.think}, "stream": False}
    for attempt in range(3):                    # one failed request must not end the whole run
        try:
            t = time.perf_counter()
            out = s.post(args.llm + "/v1/chat/completions", json=body, timeout=900).json()
            ms = (time.perf_counter() - t) * 1000
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(10)
    msg = out["choices"][0]["message"]
    try:
        choice = json.loads(msg["content"])["choice"]
    except Exception:
        choice = None
    return {"i": i, "kind": r["kind"], "label": r["label"], "choice": choice, "right": choice == r["label"],
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "tokens": (out.get("usage") or {}).get("completion_tokens"), "ms": round(ms)}


ok = {"command": 0, "weapon": 0}
tot = {"command": 0, "weapon": 0}
lat = []
done = []
if args.resume and args.dump and os.path.exists(args.dump):
    done = [json.loads(l) for l in open(args.dump)]
    have = {r["i"] for r in done}
    rows = [(i, r) for i, r in rows if i not in have]
    for row in done:
        lat.append(row["ms"])
        tot[row["kind"]] += 1
        ok[row["kind"]] += row["right"]
    print(f"resume: {len(done)} rows kept, {len(rows)} to ask")
dump = open(args.dump, "a" if done else "w") if args.dump else None
with ThreadPoolExecutor(args.workers) as pool:
    for row in pool.map(ask, rows):
        lat.append(row["ms"])
        tot[row["kind"]] += 1
        ok[row["kind"]] += row["right"]
        if dump:
            dump.write(json.dumps(row) + "\n")
            dump.flush()
for k in ("command", "weapon"):
    if tot[k]:
        print(f"{k:8} {ok[k]}/{tot[k]} = {ok[k] / tot[k]:.3f}")
lat.sort()
print(f"{args.label}: latency p50 {lat[len(lat) // 2]:.0f} ms, think={args.think}")
