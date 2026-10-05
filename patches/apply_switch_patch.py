"""Adds --send-frame to doomLaya's agent.py (on top of doomLaya-agent.patch).

With --send-frame every decision request also carries `meta` (game time, position,
health, kills, map, last execution status) and `image` (the game frame as a
320x240 JPEG data URL). The actor bridges ignore the extra fields; switch.py uses
them to notice when the player is stuck and to show the frame to the thinker.

Idempotent: running it twice changes nothing. Usage: python apply_switch_patch.py path/to/agent.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "agent.py")
s = p.read_text()
if "--send-frame" in s:
    print("already patched")
    raise SystemExit(0)

EDITS = [
    ("    def predict(self, text, questions=None):\n        questions=questions or QUESTIONS\n        started=time.perf_counter()\n"
     "        response=self.session.post(self.endpoint,json={'state':text,'questions':questions,'model':self.model},timeout=(3,15))",
     "    def predict(self, text, questions=None, extra=None):\n        questions=questions or QUESTIONS\n        started=time.perf_counter()\n"
     "        response=self.session.post(self.endpoint,json={'state':text,'questions':questions,'model':self.model,**(extra or {})},timeout=(3,15))"),
    ("    def predict(self,text,questions=None):\n        questions=questions or QUESTIONS\n        started=time.perf_counter()\n"
     "        response=self.session.post(self.endpoint,json={'model':'typesafe/jev-1.13'",
     "    def predict(self,text,questions=None,extra=None):\n        questions=questions or QUESTIONS\n        started=time.perf_counter()\n"
     "        response=self.session.post(self.endpoint,json={'model':'typesafe/jev-1.13'"),
    ("    parser.add_argument('--interval', type=float, default=.5,",
     "    parser.add_argument('--send-frame', action='store_true', help='also send game meta and a 320x240 frame (for switch.py)')\n"
     "    parser.add_argument('--interval', type=float, default=.5,"),
    ("                else:future=pool.submit(client.predict,text,packet['questions'])",
     "                else:\n"
     "                    extra=None\n"
     "                    if args.send_frame:\n"
     "                        import base64, io\n"
     "                        from PIL import Image\n"
     "                        buf=io.BytesIO()\n"
     "                        Image.fromarray(raw.screen_buffer).resize((320,240)).save(buf,'JPEG',quality=70)\n"
     "                        ex=s.get('execution',{})\n"
     "                        extra={'meta':{'game_seconds':round(tick/TICRATE,3),'tick':tick,'x':float(s['x']),'y':float(s['y']),\n"
     "                                       'hp':float(s['hp']),'kills':int(s['kills']),'map':current_map,\n"
     "                                       'execution':ex.get('status'),'detail':ex.get('detail')},\n"
     "                               'image':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}\n"
     "                    future=pool.submit(client.predict,text,packet['questions'],extra)"),
]
for old, new in EDITS:
    if s.count(old) != 1:
        raise SystemExit(f"anchor not found exactly once: {old[:70]!r}")
    s = s.replace(old, new)
p.with_suffix(".py.bak_pre_switch").write_text(p.read_text())
p.write_text(s)
print("patched", p)
