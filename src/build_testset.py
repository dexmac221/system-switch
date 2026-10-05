"""A larger held-out test set, built EXACTLY like doomLaya's validation set.

doomLaya's training/build_dataset.py labels game states with rule-based expert
labels (`gold`); its validation split is capped at 25 questions per
(kind, category) group, i.e. 130 questions, too few for confidence intervals.
This script reuses the same pipeline (policy.request, Executor memory replay,
the same `gold` function imported from build_dataset.py) on OTHER recorded games:
MAP01 runs played with seeds that doomLaya never used (its train/validation runs
were seeds 42, 43 and 44), by any player, and with a larger cap per group.
No synthetic variants: only states that actually happened.

Run from the doomLaya directory:
  build_testset.py --runs runs/*_s48 runs/*_s49 ... --cap 150 --out training/test_large.json
"""
import argparse
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", nargs="+", required=True, help="run directories (with telemetry.jsonl)")
ap.add_argument("--cap", type=int, default=150, help="max questions per (kind, category) group")
ap.add_argument("--every", type=int, default=14, help="label one state every N tics (doomLaya: 14)")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=7081)
args = ap.parse_args()

root = Path.cwd()
sys.path.insert(0, str(root))
from policy import request          # noqa: E402  (doomLaya modules)
from executor import Executor       # noqa: E402
from mission import Mission, map_data  # noqa: E402
import vizdoom                      # noqa: E402

# import gold() from doomLaya's own builder without running its main loop
src = (root / "training" / "build_dataset.py").read_text()
gold_src = src[src.index("def gold("):src.index("\nfor split,names in paths.items():")]
ns = {"utility": __import__("items").utility}
exec(compile(gold_src, "build_dataset.gold", "exec"), ns)
gold = ns["gold"]

mission = Mission(map_data(Path(vizdoom.__file__).parent / "freedoom2.wad", "MAP01"))
random.seed(args.seed)
groups = collections.defaultdict(list)
used = []
for run in map(Path, args.runs):
    tel = run / "telemetry.jsonl"
    if not tel.exists():
        continue
    rows = [json.loads(l) for l in tel.read_text().splitlines()]
    if not any(r.get("map") == "MAP01" for r in rows):
        continue
    used.append(run.name)
    memory = Executor()
    episode = None
    for row in rows:
        if episode != row["episode"]:
            memory = Executor()
            episode = row["episode"]
        memory.observe(row, row["tick"])
        if row["tick"] % args.every or row.get("map") != "MAP01":
            continue
        row.pop("execution", None)
        row.pop("target_failures", None)
        packet = request(row, memory.known, mission)
        labels = gold(packet, row)
        for key, q in packet["questions"].items():
            choice = labels[key]
            category = packet["commands"][choice]["action"] if key == "command" else choice
            groups[(key, category)].append({"state": packet["state"], "question": q, "label": choice, "kind": key,
                                            "category": category, "source_run": run.name,
                                            "source_tick": row["tick"], "synthetic": False})
records = []
for group, values in sorted(groups.items()):
    random.shuffle(values)
    records += values[:args.cap]
random.shuffle(records)
Path(args.out).write_text(json.dumps(records, indent=1))
print(f"{len(used)} runs, {len(records)} questions ->", args.out)
print(dict(collections.Counter((r["kind"], r["category"]) for r in records)))
