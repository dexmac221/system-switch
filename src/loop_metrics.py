"""Closed-loop measures beyond exits, for games where nobody exits: area covered (distinct
128-unit cells, the same grid as the switch's novelty trigger), closest approach to the exit
(meters, 32 map units = 1 m as in doomLaya), doors opened, deaths; and, from the switch log,
how much of the game the thinker was busy and how old the frame was when its plan started.

Run from the doomLaya directory: loop_metrics.py <tag prefix, e.g. g5-> [--logs switch/logs]
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prefix")
ap.add_argument("--logs", default="switch/logs")
ap.add_argument("--cell", type=float, default=128.0)
args = ap.parse_args()

sys.path.insert(0, str(Path.cwd()))
import vizdoom                                    # noqa: E402
from mission import Mission, map_data             # noqa: E402

exit_center = Mission(map_data(Path(vizdoom.__file__).parent / "freedoom2.wad", "MAP01")).exit["center"]
ex, ey = (exit_center["x"], exit_center["y"]) if isinstance(exit_center, dict) else exit_center

rows = defaultdict(list)
for run in sorted(Path("runs").glob(f"*_{args.prefix}*")):
    m = re.search(re.escape(args.prefix) + r"(.+)-s(\d+)$", run.name)
    if not m or not (run / "summary.json").exists():
        continue
    s = json.loads((run / "summary.json").read_text())
    tel = [json.loads(l) for l in open(run / "telemetry.jsonl")]
    cells = {(math.floor(t["x"] / args.cell), math.floor(t["y"] / args.cell)) for t in tel}
    closest = min(math.hypot(t["x"] - ex, t["y"] - ey) for t in tel) / 32
    g = {"seed": int(m.group(2)), "cells": len(cells), "closest_m": closest, "doors": s.get("doors_opened"),
         "deaths": s["deaths"], "wall_s": s["wall_seconds"], "game_s": s["game_seconds"]}
    log = Path(args.logs) / f"switch_{run.name.split('_', 3)[-1]}.jsonl"
    if log.exists():
        th = [r for r in map(json.loads, open(log)) if r["kind"] == "thought" and r.get("thought")]
        g["busy_share"] = sum(r["latency_ms"] for r in th) / 1000 / s["game_seconds"]
        g["staleness_s"] = sorted(r["latency_ms"] / 1000 for r in th)[len(th) // 2] if th else None
    rows[m.group(1)].append(g)


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


print(f"exit at ({ex:.0f}, {ey:.0f})")
print(f"{'variant':10} {'n':>2} {'cells':>6} {'closest m':>9} {'doors':>6} {'deaths':>6} {'busy':>6} {'stale s':>7} {'wall/game':>9}")
for v, gs in rows.items():
    print(f"{v:10} {len(gs):>2} {mean([g['cells'] for g in gs]):6.1f} {mean([g['closest_m'] for g in gs]):9.1f} "
          f"{mean([g['doors'] for g in gs]):6.1f} {mean([g['deaths'] for g in gs]):6.1f} "
          f"{mean([g.get('busy_share') for g in gs]):6.2f} {mean([g.get('staleness_s') for g in gs]):7.1f} "
          f"{mean([g['wall_s'] / g['game_s'] for g in gs]):9.2f}")
    for g in sorted(gs, key=lambda g: g["seed"]):
        print(f"   s{g['seed']}: cells {g['cells']}, closest {g['closest_m']:.1f} m, doors {g['doors']}, deaths {g['deaths']}")
