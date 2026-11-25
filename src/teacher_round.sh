#!/bin/bash
# 13 ott 2026 — System Switch v2: un GIRO di raccolta col maestro. Laya base (allievo) resta accesa sulla 4070; per ogni
# seme un switch.py nuovo (la memoria del cancello non passa da una partita all'altra) e una partita con agent.py
# --wait-model --realtime 0. Si ferma a <obiettivo> etichette, dopo <max> partite, o subito se il tunnel verso
# l'Ollama del maestro non risponde. Dopo ogni partita una riga in switch/giri/<giro>/progresso.log, spesa compresa
# (dai token veri registrati in ogni etichetta). Video solo della prima partita (75 MB ogni 40 s).
# Uso: switch/maestro_giro.sh <giro> <modello maestro> <obiettivo etichette> <primo seme> [secondi=180] [mappa=MAP01] [max=40]
#      [allieva=base | <checkpoint di una Laya adattata>]   (14 ott: allieva diversa da Laya base = giro DAgger;
#      la Laya adattata sta su :8001 perché agent.py/serve_doom_laya la vogliono lì, mai su :8000)
set -u
cd ~/src/doomLaya
GIRO=$1; MAESTRO=$2; OBIETTIVO=$3; SEME=$4; SECS=${5:-180}; MAPPA=${6:-MAP01}; MAX=${7:-40}; ALLIEVA=${8:-base}
if [ "$ALLIEVA" = base ]; then AMOD=typed-decisions; APORT=8000; AARG=""
else AMOD=doom-adapted; APORT=8001; AARG="--checkpoint $ALLIEVA"; fi
DIR=switch/giri/$GIRO; mkdir -p $DIR
PROG=$DIR/progresso.log
echo "$(date '+%F %T') giro $GIRO: maestro $MAESTRO, obiettivo $OBIETTIVO etichette, semi da $SEME, $SECS s, $MAPPA, allieva $ALLIEVA" | tee -a $PROG
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python serve_doom_laya.py --model $AMOD $AARG \
    --device cuda --port $APORT > $DIR/laya.out 2>&1 &
laya=$!
trap 'kill $laya 2>/dev/null; [ -n "${sw:-}" ] && kill $sw 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -s -m 2 127.0.0.1:$APORT/health | grep -q '"ok"' && break; sleep 1; done
n=0
while [ $n -lt $MAX ]; do
    if ! curl -s -m 5 http://127.0.0.1:11435/api/version >/dev/null; then
        echo "$(date '+%F %T') STOP: il tunnel verso l'Ollama del maestro (:11435) non risponde" | tee -a $PROG; exit 2
    fi
    s=$((SEME + n))
    .venv/bin/python switch/switch.py --actor http://127.0.0.1:$APORT/predict --actor-model $AMOD --mode maestro \
        --maestro http://127.0.0.1:11435 --maestro-model "$MAESTRO" --conf 0.3 --basics --lang en --seme-sorpresa $s \
        --etichette $DIR/etichette_s$s.jsonl --fotogrammi $DIR/fotogrammi_s$s --log $DIR/switch_s$s.jsonl \
        --port 8010 > $DIR/switch_s$s.out 2>&1 &
    sw=$!
    for i in $(seq 1 40); do curl -s -m 2 127.0.0.1:8010/health | grep -q '"ok"' && break; sleep 0.5; done
    REC=""; [ $n -eq 0 ] && REC="--record auto"
    t0=$(date +%s)
    timeout $((SECS * 30 + 300)) .venv/bin/python agent.py --model llm --endpoint http://127.0.0.1:8010/predict \
        --send-frame --wait-model --realtime 0 --map $MAPPA --seed $s --skill 3 --seconds $SECS --stop-after-level $REC \
        --tag $GIRO-s$s > $DIR/agent_s$s.out 2>&1
    kill $sw 2>/dev/null; wait $sw 2>/dev/null; sw=""
    run=$(grep -o "RUN [^ ]*" $DIR/agent_s$s.out | head -1 | cut -d" " -f2)
    riga=$(.venv/bin/python - "$run" "$DIR" "$s" "$MAPPA" "$MAESTRO" "$(( $(date +%s) - t0 ))" <<'PY'
import glob, json, os, sys
run, d, seme, mappa, maestro, orologio = sys.argv[1:7]
PREZZI = {"gemma4": (0.14, 0.40), "deepseek-v4.1-flash": (0.30, 1.20)}   # $/Mtoken, ingresso/uscita (ore di punta)
p = next((v for k, v in PREZZI.items() if maestro.startswith(k)), (0.30, 1.20))
s = json.load(open(os.path.join(run, "summary.json"))) if run and os.path.exists(os.path.join(run, "summary.json")) else {}
ev = [json.loads(l) for l in open(os.path.join(run, "events.jsonl"))] if run and os.path.exists(os.path.join(run, "events.jsonl")) else []
esce = next((e["game_seconds"] for e in ev if e.get("event") == "level_finished" and e.get("map") == mappa), None)
qui = [json.loads(l) for l in open(os.path.join(d, f"etichette_s{seme}.jsonl"))] if os.path.exists(os.path.join(d, f"etichette_s{seme}.jsonl")) else []
tutte = [json.loads(l) for f in glob.glob(os.path.join(d, "etichette_s*.jsonl")) for l in open(f)]
tin = sum(r.get("token_in", 0) for r in tutte); tout = sum(r.get("token_out", 0) for r in tutte)
spesa = tin * p[0] / 1e6 + tout * p[1] / 1e6
valide = sum(1 for r in qui if r.get("maestro"))
print(f"seme {seme}: esce {esce if esce is not None else 'no'} | uccisioni {s.get('kills')} morti {s.get('deaths')} | "
      f"decisioni {s.get('decisions')} | etichette {len(qui)} (valide {valide}) | orologio {orologio} s | "
      f"TOTALE etichette {len(tutte)} | token {tin}/{tout} | spesa ${spesa:.3f}")
print(len(tutte))
PY
)
    echo "$(date '+%F %T') $(echo "$riga" | head -1)" | tee -a $PROG
    tot=$(echo "$riga" | tail -1)
    n=$((n + 1))
    [ "$tot" -ge "$OBIETTIVO" ] 2>/dev/null && break
done
echo "$(date '+%F %T') giro $GIRO finito dopo $n partite" | tee -a $PROG
