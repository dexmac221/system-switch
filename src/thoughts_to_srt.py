"""Subtitles of the switch's thoughts for the game video.

doomLaya's recorder writes one frame per game tic at 35 fps, so video time = game
time, and the thoughts in switch.py's log are stamped in game seconds. A thought
appears when it ARRIVES (the call time plus the thinker's latency: thinking takes
time, and the video should show that) and stays for its plan, at least 3 s.

Usage: thoughts_to_srt.py switch_log.jsonl > thoughts.srt
       ffmpeg -i video.mp4 -vf "subtitles=thoughts.srt:force_style='FontSize=22'" -c:a copy video_thoughts.mp4
"""
import json
import sys


def stamp(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


rows = [json.loads(l) for l in open(sys.argv[1])]
thoughts = [r for r in rows if r.get("kind") == "thought" and r.get("thought")]
for i, r in enumerate(thoughts, 1):
    start = r.get("plan_from") or (r["t"] + r.get("latency_ms", 0) / 1000)
    end = start + max(3, r.get("hold_seconds") or 3)
    if i < len(thoughts):
        nxt = thoughts[i].get("plan_from") or (thoughts[i]["t"] + thoughts[i].get("latency_ms", 0) / 1000)
        end = min(end, nxt - 0.05)
    print(f"{i}\n{stamp(start)} --> {stamp(end)}\n💭 {r['thought']}\n")
