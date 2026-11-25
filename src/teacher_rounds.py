"""The teacher's labelling rounds (System Switch II, Section 7): MAP01, seeds 100-111, 180 s of game time, the game
paused while the teacher answers.

Reads data/teacher/rounds/<round>/etichette_s<seed>.jsonl (one line per label) and progresso.log (one line per game).
Fields of a label (original names): tipo = "cancello" (the gate was open: the teacher's choice is executed) or
"sorpresa" (surprise check with the gate closed: the student's choice is executed); perche = the gate's reasons;
allievo / conf = the student's choice and top probability; maestro / pensiero = the teacher's choice and reason;
maestro_modello, token_in, token_out, latency_ms; state and questions = what the student was asked.

Per round: games that reach the exit, deaths, labels, cost; labels by kind with the student-teacher agreement and
the student's median confidence; the gate's reasons; command types chosen by teacher and student; how often the
teacher changes command type between two consecutive labels (within 0.6 s of game time).

Usage: teacher_rounds.py data/teacher/rounds
"""
import collections, glob, json, re, statistics as st, sys

G = sys.argv[1]
ROUNDS = [("gemma1", "Gemma 4 31B, no memory of its own decisions; student: zero-shot"),
          ("gemma2", "Gemma 4 31B; student: zero-shot"),
          ("deepseek1", "DeepSeek V4.1 Flash; student: zero-shot"),
          ("dagger1", "Gemma 4 31B; student: trained on gemma2 (run 1)")]
ctype = lambda c: "none" if not c else ("open_door" if c.startswith("open") else re.sub(r"_\d+$", "", c))
reason = lambda p: p.split(":")[0]

for g, desc in ROUNDS:
    files = sorted(glob.glob(f"{G}/{g}/etichette_s*.jsonl"))
    per_game = [[json.loads(l) for l in open(f) if l.strip()] for f in files]
    R = [r for game in per_game for r in game]; V = [r for r in R if r.get("maestro")]
    log = open(f"{G}/{g}/progresso.log").read()
    games = re.findall(r"seme (\d+): esce (\S+) \| uccisioni (\d+) morti (\d+) \| decisioni (\d+)", log)
    cost = re.findall(r"spesa \$([\d.]+)", log)
    print(f"== {g}: {desc}; teacher model {dict(collections.Counter(r['maestro_modello'] for r in V))}")
    print(f"   games {len(games)}, reach the exit {sum(e != 'no' for _, e, *_ in games)}, deaths {sum(int(x[3]) for x in games)}, "
          f"decisions {sum(int(x[4]) for x in games)}; labels {len(R)} (valid {len(V)}), {len(V) / sum(int(x[4]) for x in games):.0%} of the decisions; "
          f"cost ${cost[-1]}; teacher latency median {st.median(int(r['latency_ms']) for r in V) / 1000:.1f} s; "
          f"sees the frame in {sum(str(r.get('vede')) == 'True' for r in V) / len(V):.0%} of the labels")
    for kind, name in (("cancello", "gate open"), ("sorpresa", "surprise check")):
        S = [r for r in V if r["tipo"] == kind]
        print(f"   {name:15} {len(S):5} ({len(S) / len(V):.0%})  student agrees with teacher {sum(r['allievo'] == r['maestro'] for r in S) / len(S):.0%} "
              f"(same command type {sum(ctype(r['allievo']) == ctype(r['maestro']) for r in S) / len(S):.0%}); "
              f"student's median confidence {st.median(float(r['conf']) for r in S):.2f}")
    for kind, name in (("cancello", "gate open"), ("sorpresa", "surprise check")):
        S = [r for r in V if r["tipo"] == kind]
        pos = [float(r["conf"]) for r in S if r["allievo"] == r["maestro"]]; neg = [float(r["conf"]) for r in S if r["allievo"] != r["maestro"]]
        au = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))
        srt = sorted(S, key=lambda r: float(r["conf"])); t = len(srt)
        terc = [sum(r["allievo"] == r["maestro"] for r in srt[i * t // 3:(i + 1) * t // 3]) / len(srt[i * t // 3:(i + 1) * t // 3]) for i in range(3)]
        print(f"   {name:15} AUROC of the student's confidence for agreeing with the teacher {au:.2f}; confidence below 0.3 on "
              f"{sum(float(r['conf']) < 0.3 for r in S) / len(S):.0%}; mean confidence {st.mean(float(r['conf']) for r in S):.2f}; "
              f"agreement by confidence tercile {terc[0]:.0%}, {terc[1]:.0%}, {terc[2]:.0%}")
    gate = [r for r in V if r["tipo"] == "cancello"]; m = collections.Counter()
    for r in gate:
        for p in {reason(p) for p in r["perche"]}:
            m[p] += 1
    print("   gate reasons (a label can have several): " + ", ".join(f"{k} {v} ({v / len(gate):.0%})" for k, v in m.most_common()))
    for who, key in (("teacher", "maestro"), ("student", "allievo")):
        c = collections.Counter(ctype(r[key]) for r in V)
        print(f"   {who} chooses: " + ", ".join(f"{k} {v / len(V):.0%}" for k, v in c.most_common()))
    pairs = changes = 0
    for game in per_game:
        r = [x for x in game if x.get("maestro")]
        for a, b in zip(r, r[1:]):
            if b["t"] - a["t"] <= 0.6:
                pairs += 1; changes += ctype(a["maestro"]) != ctype(b["maestro"])
    print(f"   teacher changes command type between consecutive labels: {changes}/{pairs} = {changes / max(1, pairs):.0%}")
    ex = collections.Counter(str(r.get("execution")) for r in V)
    print(f"   executor report at the labelled states: " + ", ".join(f"{k} {v / len(V):.0%}" for k, v in ex.most_common(4)))
    tr = [r for f, game in zip(files, per_game) for r in game if r.get("maestro") and int(re.search(r"_s(\d+)", f).group(1)) < 110]
    print(f"   labels of seeds 100-109 (training): {len(tr)}, of which surprise checks {sum(r['tipo'] == 'sorpresa' for r in tr) / len(tr):.0%}; "
          f"labels of seeds 110-111 (held out): {len(V) - len(tr)}")
