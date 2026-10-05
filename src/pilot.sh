#!/bin/bash
# pilot.sh <tag> <actor /predict URL> <actor model> <mode> <map> <seed> [switch.py args...]
# One game through switch.py: starts the switch, plays (agent.py --send-frame, video), stops
# the switch, prints one JSON line (exit, kills, deaths, thoughts, plan share) and keeps
# switch_<tag>.jsonl next to the run. Run it from the doomLaya directory.
set -u
tag=$1; actor=$2; amodel=$3; mode=$4; map=$5; seed=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-.venv/bin/python}; SECS=${GAME_SECONDS:-180}; PORT=${PORT_SWITCH:-8010}
LOG=$HERE/logs/switch_${tag}.jsonl; rm -f "$LOG"
"$PY" "$HERE/switch.py" --actor "$actor" --actor-model "$amodel" --mode "$mode" --log "$LOG" --port $PORT "$@" \
      > "$HERE/logs/switch_${tag}.out" 2>&1 &
sw=$!
for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 0.5; done
out=$(timeout $((SECS + 120)) "$PY" agent.py --model llm --endpoint http://127.0.0.1:$PORT/predict --send-frame \
      --map "$map" --seed "$seed" --skill 3 --seconds "$SECS" --stop-after-level --record auto --tag "$tag" 2>&1)
kill $sw 2>/dev/null; wait $sw 2>/dev/null
run=$(echo "$out" | grep -o "RUN [^ ]*" | head -1 | cut -d" " -f2)
"$PY" - "$run" "$tag" "$map" "$seed" "$LOG" "$mode" <<'PY'
import json, os, sys
run, tag, game_map, seed, log, mode = sys.argv[1:7]
s = json.load(open(os.path.join(run, "summary.json")))
ev = [json.loads(l) for l in open(os.path.join(run, "events.jsonl"))]
exit_s = next((e["game_seconds"] for e in ev if e.get("event") == "level_finished" and e.get("map") == game_map), None)
rows = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
req = [r for r in rows if r["kind"] == "request"]
th = [r for r in rows if r["kind"] == "thought"]
print(json.dumps({"tag": tag, "mode": mode, "map": game_map, "seed": int(seed), "exit_s": exit_s,
                  "kills": s["kills"], "deaths": s["deaths"], "errors": s["errors"], "decisions": s["decisions"],
                  "thoughts": len([t for t in th if t.get("thought")]), "thinker_errors": len([t for t in th if t.get("error")]),
                  "plan_share": round(sum(r["source"] == "plan" for r in req) / max(1, len(req)), 3),
                  "thinker_ms_p50": sorted(t["latency_ms"] for t in th if t.get("thought"))[len([t for t in th if t.get("thought")]) // 2] if any(t.get("thought") for t in th) else None,
                  "run": run}), flush=True)
PY
