"""Summary of closed-loop games by condition, from the JSON lines that pilot.sh prints.
Tags look like <prefix>-<condition>-s<seed> (e.g. i5-unsure-s48).

Usage: summarize_games.py games.out [more.out ...]
"""
import json
import re
import sys
from collections import defaultdict

games = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("{"):
            continue
        g = json.loads(line)
        m = re.match(r"(?:[^-]+-)?(.+)-s(\d+)$", g["tag"])
        games[m.group(1) if m else g["tag"]].append(g)


def med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


print(f"{'condition':12} {'games':>5} {'exits':>5} {'exit s':>12} {'kills':>6} {'deaths':>6} "
      f"{'thoughts':>8} {'plan share':>10} {'think p50 s':>11}")
for cond, gs in games.items():
    exits = [g["exit_s"] for g in gs if g["exit_s"] is not None]
    lat = med([g["thinker_ms_p50"] for g in gs])
    print(f"{cond:12} {len(gs):>5} {len(exits):>5} {','.join(f'{x:.0f}' for x in exits) or '-':>12} "
          f"{sum(g['kills'] for g in gs) / len(gs):6.1f} {sum(g['deaths'] for g in gs) / len(gs):6.1f} "
          f"{sum(g['thoughts'] for g in gs) / len(gs):8.1f} {sum(g['plan_share'] for g in gs) / len(gs):10.2f} "
          f"{lat / 1000 if lat else 0:11.1f}")
    for g in gs:
        print(f"   s{g['seed']}: exit {g['exit_s']}, kills {g['kills']}, deaths {g['deaths']}, "
              f"thoughts {g['thoughts']} (errors {g['thinker_errors']}), plan {g['plan_share']}, "
              f"decisions {g['decisions']}")
