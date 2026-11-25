#!/usr/bin/env python3
"""13 ott 2026 — System Switch v2: dalle etichette del maestro ai dati di training/finetune.py.

Ogni etichetta valida diventa una riga {state, question, label, kind: "command"}: lo stato che ha visto l'allievo, la
domanda «command» con i comandi di quel momento, e la scelta del MAESTRO. La validazione si separa PER PARTITA (per
seme), come in training/datasets.json: righe vicine della stessa partita si somigliano, e mescolarle gonfierebbe
l'accuratezza. Il validation.json originale NON serve qui: misura l'accordo con l'oracolo a regole, non col maestro.

Uso: etichette_a_training.py <cartella del giro> [<altra cartella> ...] --out <cartella> [--val-semi 108,109]
     [--tetto-tipo 0.45]   (nessun tipo di comando oltre questa quota del training: il maestro raccoglie volentieri)
"""
import argparse, collections, glob, json, os, random, re

ap = argparse.ArgumentParser()
ap.add_argument("giri", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--val-semi", default=None, help="semi tenuti per la validazione (predefinito: gli ultimi due)")
ap.add_argument("--tetto-tipo", type=float, default=None, help="quota massima di un tipo di comando nel training")
ap.add_argument("--seme", type=int, default=771)
a = ap.parse_args()

righe = []
for g in a.giri:
    for f in sorted(glob.glob(os.path.join(g, "etichette_s*.jsonl"))):
        seme = int(re.search(r"_s(\d+)\.jsonl$", f).group(1))
        for n, l in enumerate(open(f, encoding="utf-8")):
            e = json.loads(l)
            q = (e.get("questions") or {}).get("command")
            if not e.get("maestro") or not q or e["maestro"] not in q.get("criteria", {}):
                continue
            righe.append({"state": e["state"], "question": q, "label": e["maestro"], "kind": "command",
                          "category": e["maestro"].split("_")[0], "run": f"{os.path.basename(g.rstrip('/'))}_s{seme}",
                          "tick": round(e["t"] * 35), "synthetic": False, "seme": seme, "tipo": e["tipo"],
                          "allievo": e["allievo"], "maestro_modello": e.get("maestro_modello")})
semi = sorted({r["seme"] for r in righe})
val = set(int(x) for x in a.val_semi.split(",")) if a.val_semi else set(semi[-2:])
train = [r for r in righe if r["seme"] not in val]
valid = [r for r in righe if r["seme"] in val]
tolte = 0
if a.tetto_tipo:
    rng = random.Random(a.seme)
    conta = collections.Counter(r["category"] for r in train)
    tetto = int(a.tetto_tipo * len(train))
    for cat, n in conta.items():
        if n > tetto:
            rng.shuffle(idx := [i for i, r in enumerate(train) if r["category"] == cat])
            via = set(idx[tetto:])
            tolte += len(via)
            train = [r for i, r in enumerate(train) if i not in via]
os.makedirs(a.out, exist_ok=True)
json.dump(train, open(os.path.join(a.out, "train.json"), "w"), ensure_ascii=False)
json.dump(valid, open(os.path.join(a.out, "validation.json"), "w"), ensure_ascii=False)
info = {"giri": a.giri, "semi": semi, "val_semi": sorted(val), "train": len(train), "validation": len(valid),
        "tolte_per_tetto": tolte, "tetto_tipo": a.tetto_tipo,
        "train_tipi": collections.Counter(r["category"] for r in train).most_common(),
        "val_tipi": collections.Counter(r["category"] for r in valid).most_common(),
        "accordo_allievo_maestro_val": round(sum(r["allievo"] == r["label"] for r in valid) / max(1, len(valid)), 3)}
json.dump(info, open(os.path.join(a.out, "info.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False))
