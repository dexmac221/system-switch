"""One player, one switch: a fast actor plays, and when acting stops working the
system switches to thinking.

dexmac's description of how he plays: while the action works the brain "switches
off" and just acts, without talking; when shooting no longer helps (a dilemma, a
door that does not open) he decides to think, and thinking is what gets him
moving again instead of shooting or wandering at random. So this is not
"System 1 + System 2" as two separate players, but one system with a switch.

This server sits between doomLaya's agent.py and the actor, with the same
/predict interface:

  agent.py --model llm --endpoint http://127.0.0.1:8010/predict --send-frame
      |
      v
  switch.py :8010 --> ACTOR (any /predict bridge: serve_llm.py, serve_systemone.py, doomLaya)
      |                 answers every request (every 0.5 s of game time)
      |
      +-- GATE: should we stop and think?  rules on the recent history (stuck,
      |         blocked commands, low confidence) and/or a System One model
      |         (noul question via llama.cpp /v1/systemone)
      |
      +-- THINKER: a vision LLM (OpenAI-compatible, e.g. llama-server --mmproj)
                gets the game frame, the situation, the commands and the
                history; writes one first-person thought and picks a command,
                which then overrides the actor for a few seconds ("switch" mode)

Modes: off = pass-through with logging (baseline), narrate = the thinker only
comments (never overrides; it also speaks on damage and kills), switch = the
thinker comments and its choice is followed for `hold_seconds`.
advise = the thought is appended to the actor's state ("Advice from your
deliberate mind: ...") while the plan lasts, and the actor still decides.
rule = the honest control: same gate and timing, but instead of thinking it
follows a fixed command (default explore). If `rule` does as well as `switch`,
the thinking adds nothing.

Every request and every thought goes to --log (jsonl); thoughts are also printed,
and thoughts_to_srt.py turns them into subtitles for the game video.

Usage:
  switch.py --actor http://127.0.0.1:8002/predict --actor-model llm \
            --thinker http://127.0.0.1:8090 --mode switch --lang it \
            --log switch_map02_s48.jsonl --port 8010
"""
import argparse
import json
import math
import re
import threading
import time
from collections import deque

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BLOCKED = re.compile(r"Previous command result: (blocked|unavailable)")
CLEAR = "The area in sight is clear of enemies"


class Switch:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.sess = requests.Session()
        self.sess.trust_env = False
        self.history = deque(maxlen=24)      # recent requests: t, command, conf, blocked, pos...
        self.positions = deque(maxlen=200)   # (t, x, y)
        self.thoughts = deque(maxlen=4)      # last thoughts, given back to the thinker as memory
        self.plan = None                     # {'choice', 'action', 'until', 'thought'}
        self.thinking = False
        self.last_thought_t = -1e9
        self.t0 = time.perf_counter()
        self.latest_t = 0.0                  # game time of the newest request
        self.log = open(args.log, "a", buffering=1) if args.log else None
        self.frame = None
        self.prev_hp = None
        self.prev_kills = None
        self.hp_window = deque(maxlen=40)    # (t, hp)
        self.cells = {}                      # grid cell -> first time visited (novelty trigger)

    # ------------------------------------------------------------------ utils
    def write(self, row):
        if self.log:
            self.log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def now(self, meta):
        if meta and meta.get("game_seconds") is not None:
            return float(meta["game_seconds"])
        return time.perf_counter() - self.t0

    def displacement(self, t, x, y, window):
        pts = [(px, py) for (pt, px, py) in self.positions if t - pt <= window]
        if len(pts) < 3 or self.positions[0][0] > t - window * 0.8:
            return None                       # not enough history yet
        return max(math.hypot(x - px, y - py) for px, py in pts)

    # ------------------------------------------------------------------- gate
    def gate(self, t, state, meta, conf, command):
        """Return the list of reasons to think now (empty = keep acting)."""
        a = self.a
        reasons = []
        recent = list(self.history)[-6:]
        if "blocked" in a.triggers and sum(h["blocked"] for h in recent) >= a.blocked_n:
            reasons.append(f"blocked: the last commands could not be executed ({sum(h['blocked'] for h in recent)}/{len(recent)})")
        if "stuck" in a.triggers and meta and meta.get("x") is not None:
            d = self.displacement(t, meta["x"], meta["y"], a.stuck_window)
            if d is not None and d < a.stuck_units and CLEAR in state:
                reasons.append(f"stuck: moved {d:.0f} map units in {a.stuck_window:.0f} s with no enemy in sight")
        low = [h for h in recent[-a.lowconf_n:]]
        if "unsure" in a.triggers and len(low) == a.lowconf_n and all(h["conf"] < a.conf for h in low):
            reasons.append(f"unsure: confidence below {a.conf} for {a.lowconf_n} decisions in a row")
        if "loop" in a.triggers:
            win = [h for h in self.history if t - h["t"] <= a.loop_window]
            if len(win) >= 8 and win[0]["t"] <= t - a.loop_window * 0.8:
                top = max(set(h["command"] for h in win), key=lambda c: sum(h["command"] == c for h in win))
                share = sum(h["command"] == top for h in win) / len(win)
                if share >= a.loop_share:
                    reasons.append(f"loop: chose {top} in {share:.0%} of the decisions of the last {a.loop_window:.0f} s")
        if "novelty" in a.triggers and meta and meta.get("x") is not None and self.cells:
            newest = max(self.cells.values())
            if t - newest >= a.novelty_window and t - min(self.cells.values()) >= a.novelty_window:
                reasons.append(f"no progress: no new area reached in {t - newest:.0f} s")
        if a.gate_sys and not reasons:
            p = self.gate_model(state)
            if p is not None and p >= a.gate_p:
                reasons.append(f"gate model: stuck or mistaken with p={p:.2f}")
        return reasons

    def gate_model(self, state):
        """Optional learned gate: a System One model answers a noul question on a
        short summary of the recent history (llama.cpp /v1/systemone)."""
        hist = "\n".join(f"t={h['t']:.1f}s chose {h['command']} (confidence {h['conf']:.2f})"
                         + (" -> could not be executed" if h["blocked"] else "") for h in list(self.history)[-8:])
        body = {"state": state + "\n\nRecent decisions:\n" + hist,
                "questions": {"think": {"type": "noul",
                                        "instructions": "Is the player stuck, looping or making a mistake, "
                                                        "so that stopping to think would help?"}}}
        try:
            r = self.sess.post(self.a.gate_sys + "/v1/systemone", json=body, timeout=2)
            return float(r.json()["answers"]["think"]["noul"])
        except Exception:
            return None

    def events(self, t, meta):
        """Narration-worthy events (narrate mode): damage bursts and kills."""
        out = []
        if not meta:
            return out
        hp, kills = meta.get("hp"), meta.get("kills")
        if hp is not None:
            self.hp_window.append((t, hp))
            old = [h for (tt, h) in self.hp_window if t - tt <= 2.0]
            if old and max(old) - hp >= self.a.damage:
                out.append(f"under fire: lost {max(old) - hp:.0f} health in 2 s")
        if kills is not None and self.prev_kills is not None and kills > self.prev_kills:
            out.append("enemy killed")
        self.prev_kills = kills
        return out

    # ---------------------------------------------------------------- thinker
    def think(self, t, reasons, state, criteria, frame):
        a = self.a
        keys = list(criteria)
        forbidden = []
        if a.forbid_repeats and any(r.split(":")[0] in ("loop", "stuck", "no progress", "blocked") for r in reasons):
            # what the actor kept choosing while not getting anywhere: the thinker must try something else
            win = [h["command"] for h in self.history if t - h["t"] <= a.loop_window]
            forbidden = [k for k in set(win) if k != "explore" and win.count(k) / max(1, len(win)) >= 0.3]
            if len([k for k in keys if k not in forbidden]) >= 1:
                keys = [k for k in keys if k not in forbidden]
            else:
                forbidden = []
        hist = "\n".join(f"t={h['t']:.1f}s {h['command']} (conf {h['conf']:.2f})"
                         + (" -> NOT executed" if h["blocked"] else "") for h in list(self.history)[-10:])
        memory = "\n".join(f"t={x['t']:.1f}s: {x['thought']}" for x in self.thoughts) or "none"
        lang = {"it": "Italian", "en": "English"}.get(a.lang, a.lang)
        system = ("You are the deliberate mind of a DOOM player. A fast reflex system is playing; it handed "
                  "control to you because acting stopped working (or something important happened). "
                  "Look at the frame and the recent history, work out WHY it is not working, then pick ONE "
                  f"of the available commands to follow for the next seconds. Write `thought` in {lang}, "
                  "first person, at most 25 words, like a player thinking aloud.")
        if a.basics:
            system += (" DOOM basics: a door that does not open usually needs a key of its colour (red, blue or "
                       "yellow card or skull) that you do not have yet: then go and look for the key elsewhere. "
                       "`explore` sends the player to the least explored area: use it when stuck, looping, or when "
                       "the key or the exit you need is not in sight. Collect items only when you need them (low "
                       "health, low ammo, a key). With no immediate threat, head for the exit.")
        user = (f"Why you were called: {'; '.join(reasons)}\n\nSituation:\n{state}\n\n"
                "Available commands:\n" + "\n".join(f"- {k}: {criteria[k]}" for k in keys)
                + (f"\n\nAlready tried and NOT working (not available now): {', '.join(forbidden)}" if forbidden else "")
                + f"\n\nRecent decisions (oldest first):\n{hist or 'none'}\n\nYour previous thoughts:\n{memory}")
        content = [{"type": "text", "text": user}]
        if frame and not a.no_image:
            content.insert(0, {"type": "image_url", "image_url": {"url": frame}})
        schema = {"type": "object",
                  "properties": {"thought": {"type": "string", "maxLength": 240},
                                 "choice": {"type": "string", "enum": keys},
                                 "hold_seconds": {"type": "integer", "minimum": 1, "maximum": 8}},
                  "required": ["thought", "choice", "hold_seconds"]}
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "max_tokens": a.think_tokens if not a.think else max(a.think_tokens, 4000), "temperature": 0.3, "stream": False,
                "response_format": {"type": "json_schema", "json_schema": {"name": "thought", "schema": schema}},
                "chat_template_kwargs": {"enable_thinking": bool(a.think)}}
        started = time.perf_counter()
        row = {"kind": "thought", "t": t, "reasons": reasons, "mode": a.mode, "forbidden": forbidden}
        try:
            r = self.sess.post(a.thinker + "/v1/chat/completions", json=body, timeout=a.think_timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content") or ""
            if not content.strip() and a.think:
                # the reasoning used up the token budget and no answer came out: answer fast instead
                # of losing the moment (logged, so the paper can count how often it happens)
                row["fallback"] = "reasoning overflow: answered without thinking"
                body["chat_template_kwargs"] = {"enable_thinking": False}
                body["max_tokens"] = a.think_tokens
                r = self.sess.post(a.thinker + "/v1/chat/completions", json=body, timeout=a.think_timeout)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content") or ""
            out = json.loads(content)
            row.update(thought=out["thought"], choice=out["choice"], hold_seconds=out["hold_seconds"],
                       latency_ms=round((time.perf_counter() - started) * 1000), usage=r.json().get("usage"))
        except Exception as e:
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}", latency_ms=round((time.perf_counter() - started) * 1000))
        with self.lock:
            self.thinking = False
            self.last_thought_t = t
            if "thought" in row:
                self.thoughts.append({"t": t, "thought": row["thought"]})
                if a.mode in ("switch", "advise"):
                    # the plan starts when the thought ARRIVES: game time moved on while thinking
                    start = self.latest_t
                    self.plan = {"choice": row["choice"], "until": start + max(row["hold_seconds"], a.min_hold),
                                 "thought": row["thought"], "t": start}
                    row["plan_from"] = start
        self.write(row)
        if "thought" in row:
            print(f"💭 [{t:6.1f}s, {row['latency_ms']} ms] {row['thought']}  → {row['choice']}", flush=True)
        else:
            print(f"!! thinker error at {t:.1f}s: {row['error']}", flush=True)

    # ---------------------------------------------------------------- predict
    def predict(self, req):
        a = self.a
        meta = req.meta or {}
        t = self.now(meta)
        self.latest_t = t
        if req.image:
            self.frame = req.image
        state_for_actor = req.state
        with self.lock:
            if a.mode == "advise" and self.plan and self.latest_t <= self.plan["until"]:
                state_for_actor = req.state + "\nAdvice from your deliberate mind: " + self.plan["thought"]
        fwd = {"state": state_for_actor, "questions": req.questions, "model": a.actor_model}
        if req.image:
            fwd["image"] = req.image          # a vision actor (serve_systemone.py --vision) can use it
        r = self.sess.post(a.actor, json=fwd, timeout=(3, 15))
        if r.status_code != 200:
            raise HTTPException(502, r.text[:300])
        out = r.json()
        cmd = out["answers"]["command"]
        conf = max(cmd["probabilities"].values())
        criteria = req.questions["command"]["criteria"]
        blocked = bool(BLOCKED.search(req.state))
        events = self.events(t, meta) if a.mode == "narrate" else []
        with self.lock:
            if meta.get("x") is not None:
                self.positions.append((t, meta["x"], meta["y"]))
                cell = (int(meta["x"] // a.cell), int(meta["y"] // a.cell))
                self.cells.setdefault(cell, t)
            reasons = [] if a.mode == "off" else self.gate(t, req.state, meta, conf, cmd["choice"])
            if getattr(self, "failure", None) and a.mode != "off":
                reasons = [self.failure] + reasons
                self.failure = None
            if a.mode == "narrate":
                reasons += events
            final, source = cmd["choice"], "actor"
            plan = self.plan
            if plan:
                if a.commit:
                    # commitment: keep the plan until the executor says it is done or impossible
                    # (or a real fight starts), like a player who goes for the exit until he gets there
                    ex = meta.get("execution")
                    held = t - plan["t"]
                    hits = [h for (tt, h) in self.hp_window if t - tt <= 2.0]
                    burst = meta.get("hp") is not None and hits and max(hits) - meta["hp"] >= a.damage
                    if meta.get("hp") is not None:
                        self.hp_window.append((t, meta["hp"]))
                    why = ("target gone" if plan["choice"] not in criteria else
                           "max hold" if held > a.max_hold else
                           f"executor: {ex}" if ex in ("arrived", "blocked", "unavailable") and held >= a.min_hold else
                           "damage burst: reflexes take over" if burst else None)
                else:
                    damaged = meta.get("hp") is not None and self.prev_hp is not None and meta["hp"] < self.prev_hp
                    why = ("expired" if t > plan["until"] else "target gone" if plan["choice"] not in criteria
                           else "damage: reflexes take over" if damaged else None)
                if why:
                    self.write({"kind": "plan_end", "t": t, "choice": plan["choice"], "why": why})
                    if a.commit and why.startswith("executor: ") and why != "executor: arrived":
                        # the plan failed: think again NOW, telling the thinker why (no refractory pause)
                        self.failure = f"your plan '{plan['choice']}' failed: {meta.get('detail') or why}"
                        self.last_thought_t = -1e9
                    self.plan = plan = None
                elif a.mode == "advise":
                    source = "advised"          # the actor decided, with the thought in its state
                else:
                    final, source = plan["choice"], "plan"
            start = (reasons and not self.thinking and not self.plan
                     and t - self.last_thought_t >= a.refractory)
            if start:
                self.thinking = True
            self.history.append({"t": t, "command": cmd["choice"], "conf": conf, "blocked": blocked})
            self.prev_hp = meta.get("hp", self.prev_hp)
        if start and a.mode == "rule":
            # Control without any thinking: same gate, same timing, but the "plan" is a fixed
            # rule. If this does as well as `switch`, the thinker adds nothing.
            choice = a.rule_choice if a.rule_choice in criteria else cmd["choice"]
            with self.lock:
                self.thinking = False
                self.last_thought_t = t
                self.plan = {"choice": choice, "until": t + a.rule_hold, "thought": f"(rule) {choice}", "t": t}
            self.write({"kind": "thought", "t": t, "reasons": reasons, "mode": "rule", "thought": f"(rule) {choice}",
                        "choice": choice, "hold_seconds": a.rule_hold, "latency_ms": 0, "plan_from": t})
        elif start:
            threading.Thread(target=self.think, daemon=True,
                             args=(t, reasons, req.state, criteria, self.frame)).start()
        if source == "plan":
            eps = 1e-4
            probs = {k: (1 - eps * (len(criteria) - 1) if k == final else eps) for k in criteria}
            out["answers"]["command"] = {**cmd, "choice": final, "probabilities": probs,
                                         "actor_choice": cmd["choice"], "thought": plan["thought"]}
        self.write({"kind": "request", "t": t, "actor": cmd["choice"], "conf": round(conf, 4), "final": final,
                    "source": source, "blocked": blocked, "reasons": reasons, "think_started": bool(start),
                    "x": meta.get("x"), "y": meta.get("y"), "hp": meta.get("hp")})
        out.setdefault("routing", {})["model"] = req.model
        out["routing"]["switch"] = {"mode": a.mode, "source": source}
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True, help="actor /predict URL")
    ap.add_argument("--actor-model", default="llm", help="model name the actor bridge expects")
    ap.add_argument("--thinker", default="http://127.0.0.1:8090", help="OpenAI-compatible vision LLM")
    ap.add_argument("--mode", choices=["off", "narrate", "switch", "rule", "advise"], default="switch",
                    help="rule = control: same gate, but a fixed command instead of the thinker")
    ap.add_argument("--rule-choice", default="explore", help="rule mode: command to follow when the gate fires")
    ap.add_argument("--rule-hold", type=float, default=3.0, help="rule mode: seconds to follow it")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--triggers", default="unsure,stuck,blocked,loop,novelty",
                    help="which signals open the gate: unsure (low confidence, the ReDAct-style trigger), "
                         "stuck (not moving), blocked (commands not executed), loop (same command again and "
                         "again), novelty (no new area reached); comma separated")
    ap.add_argument("--loop-window", type=float, default=15.0, help="loop: seconds of decisions looked at")
    ap.add_argument("--loop-share", type=float, default=0.8, help="loop: share of the same command")
    ap.add_argument("--novelty-window", type=float, default=15.0, help="novelty: seconds without a new grid cell")
    ap.add_argument("--cell", type=float, default=128.0, help="novelty: grid cell size in map units")
    ap.add_argument("--think", action="store_true", help="let the thinker reason before answering (slower)")
    ap.add_argument("--min-hold", type=float, default=0.0, help="minimum seconds a thought guides the actor")
    ap.add_argument("--commit", action="store_true", help="keep a plan until the executor reports arrived/blocked/unavailable (after --min-hold), a damage burst, or --max-hold")
    ap.add_argument("--max-hold", type=float, default=45.0, help="commit: longest a plan can last, seconds")
    ap.add_argument("--forbid-repeats", action="store_true", help="when stuck/looping, the actor's repeated commands are removed from the thinker's options")
    ap.add_argument("--basics", action="store_true", help="give the thinker a few DOOM basics (keys, explore, when to collect)")
    ap.add_argument("--gate-sys", default=None, help="optional llama-server with a System One model for the gate")
    ap.add_argument("--gate-p", type=float, default=0.6)
    ap.add_argument("--blocked-n", type=int, default=3, help="blocked results among the last 6 requests")
    ap.add_argument("--stuck-window", type=float, default=5.0, help="seconds")
    ap.add_argument("--stuck-units", type=float, default=48.0, help="map units (player is 32 wide)")
    ap.add_argument("--conf", type=float, default=0.6, help="low-confidence threshold")
    ap.add_argument("--lowconf-n", type=int, default=3)
    ap.add_argument("--damage", type=float, default=20.0, help="narrate: health lost in 2 s")
    ap.add_argument("--refractory", type=float, default=6.0, help="seconds of silence after a thought")
    ap.add_argument("--think-tokens", type=int, default=200)
    ap.add_argument("--think-timeout", type=float, default=20.0)
    ap.add_argument("--no-image", action="store_true", help="thinker without the frame (ablation)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    args.triggers = set(t.strip() for t in args.triggers.split(",") if t.strip())
    sw = Switch(args)
    app = FastAPI()

    class Request(BaseModel):
        state: str
        questions: dict
        model: str = "llm"
        meta: dict | None = None
        image: str | None = None

    @app.get("/health")
    def health():
        actor = sw.sess.get(args.actor.rsplit("/", 1)[0] + "/health", timeout=5).json()
        return {"status": actor.get("status"), "models": {args.actor_model: actor, "llm": "switch"},
                "switch": {"mode": args.mode, "thinker": args.thinker, "gate_sys": args.gate_sys}}

    @app.post("/predict")
    def predict(req: Request):
        return sw.predict(req)

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
