#!/bin/bash
# bench_systemone.sh <label> <model.gguf> [extra llama-server args...]
# One System One decision model, zero-shot, through llama.cpp /v1/systemone:
#   1. starts llama-server with the model (encoders like laya/julia/clef need the whole
#      prompt in one ubatch: -b/-ub 2048, otherwise llama-server ABORTS on long Doom states)
#   2. starts serve_systemone.py (doomLaya /predict bridge)
#   3. runs the 130 validation questions (evaluate.py) and, if SEEDS is set, plays them
#   4. stops everything; one JSON line per model in ${OUT:-bench_systemone.jsonl}
# Environment: LLAMA_BIN (llama-server path), DOOMLAYA (doomLaya dir with training/v3),
#   PY (python with fastapi/uvicorn/requests), SEEDS (e.g. "48 49 50"), MAP, GAME_SECONDS,
#   CUDA_VISIBLE_DEVICES (pick the GPU; use CUDA_DEVICE_ORDER=PCI_BUS_ID), PORT_SYS, PORT_BRIDGE
set -u
label=$1; gguf=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA_BIN=${LLAMA_BIN:-llama-server}; PY=${PY:-python3}; DOOMLAYA=${DOOMLAYA:-.}
PORT_SYS=${PORT_SYS:-8095}; PORT_BRIDGE=${PORT_BRIDGE:-8004}; OUT=${OUT:-$PWD/bench_systemone.jsonl}
LOGS=${LOGS:-$PWD/bench_logs}; mkdir -p "$LOGS"

if [[ "$gguf" == http* ]]; then
  # a decision server that is already running and speaks /v1/systemone (e.g. `strands-decider serve`)
  SYS_URL=$gguf; srv=""; : > "$LOGS/$label.server.log"
else
  SYS_URL=http://127.0.0.1:$PORT_SYS
  "$LLAMA_BIN" -m "$gguf" --host 127.0.0.1 --port $PORT_SYS -c 4096 -b 2048 -ub 2048 "$@" > "$LOGS/$label.server.log" 2>&1 &
  srv=$!
  for i in $(seq 1 180); do curl -s -m 2 $SYS_URL/health | grep -q '"ok"' && break; sleep 1; done
fi
"$PY" "$HERE/serve_systemone.py" --sys $SYS_URL --name "$label" --port $PORT_BRIDGE > "$LOGS/$label.bridge.log" 2>&1 &
br=$!
for i in $(seq 1 30); do curl -s -m 2 http://127.0.0.1:$PORT_BRIDGE/health >/dev/null && break; sleep 0.5; done

val=$(cd "$DOOMLAYA" && "$PY" "$HERE/evaluate.py" http://127.0.0.1:$PORT_BRIDGE/predict llm ${DUMP_DIR:+$DUMP_DIR/dump_$label.jsonl} 2>&1 | tee "$LOGS/$label.validation.txt")
games=""
if [ -n "${SEEDS:-}" ]; then
  games=$(DOOMLAYA="$DOOMLAYA" bash "$HERE/play_seeds.sh" llm http://127.0.0.1:$PORT_BRIDGE/predict "$label" $SEEDS 2>&1 | grep '^{' | tee "$LOGS/$label.games.jsonl")
fi
crashed=$(grep -c "GGML_ASSERT" "$LOGS/$label.server.log")
kill $br $srv 2>/dev/null; [ -n "$srv" ] && wait $srv 2>/dev/null

"$PY" - "$label" "$gguf" "$crashed" <<PY >> "$OUT"
import json, re, sys
label, gguf, crashed = sys.argv[1], sys.argv[2], int(sys.argv[3])
val = """$val"""
games = [json.loads(l) for l in """$games""".splitlines() if l.startswith("{")]
m = {k: re.search(k + r"\s+(\d+)/(\d+) = ([\d.]+)", val) for k in ("command", "weapon")}
lat = re.search(r"latency p50 (\d+) ms", val)
row = {"label": label, "gguf": gguf.rsplit("/", 1)[-1], "server_crashed": bool(crashed),
       "command": float(m["command"].group(3)) if m["command"] else None,
       "weapon": float(m["weapon"].group(3)) if m["weapon"] else None,
       "latency_p50_ms": int(lat.group(1)) if lat else None,
       "games": games, "exits": sum(g.get("exit_s") is not None for g in games)}
print(json.dumps(row))
PY
tail -1 "$OUT"
