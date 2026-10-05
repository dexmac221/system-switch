"""Same /predict interface as doomLaya's serve_doom_laya.py, but the decisions are
taken by any "System One" decision model served by llama.cpp's /v1/systemone
endpoint (TypeSafe-compatible API, llama.cpp PR #29818 and #29831): laya, julia-1,
lev, kev, openjev, clef.

doomLaya already asks TypeSafe-style questions (`instructions` + `criteria`), so they
are passed through unchanged, like serve_clm.py does: one bridge for every model,
same questions, same GPU, which is the point of the benchmark.

Vision track: if the agent sends `images` (data URLs of the game frame), they are
forwarded too; the model needs image support and its --mmproj (e.g. openjev).

    llama-server -m Kev-4B-Q8_0.gguf -ngl 999 --port 8090
    serve_systemone.py --sys http://127.0.0.1:8090 --name kev-4b --port 8004
"""
import argparse
import time

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sys", default="http://127.0.0.1:8090", help="llama-server with a decision model")
    ap.add_argument("--name", required=True, help="model label in the results")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--vision", action="store_true",
                    help="forward the game frame (agent.py --send-frame sends it as `image`) to the model")
    args = ap.parse_args()
    sess = requests.Session()
    app = FastAPI()

    class Request(BaseModel):
        state: str
        questions: dict
        model: str = "llm"
        images: list[str] | None = None
        image: str | None = None             # what agent.py --send-frame sends
        meta: dict | None = None             # ignored here (used by switch.py)

    @app.get("/health")
    def health():
        try:
            props = sess.get(args.sys + "/props", timeout=5).json()
            served = props.get("model_alias") or props.get("model_path")
        except requests.RequestException as e:
            served = f"unreachable: {e}"
        return {"status": "ok", "models": {"llm": served}, "sys": args.sys, "name": args.name}

    @app.post("/predict")
    def predict(req: Request):
        t0 = time.perf_counter()
        questions = {k: q for k, q in req.questions.items() if q.get("type") == "choice"}
        body = {"state": req.state, "questions": questions}
        images = (req.images or []) + ([req.image] if req.image else [])
        if args.vision and images:
            body["images"] = images
        r = sess.post(args.sys + "/v1/systemone", json=body, timeout=args.timeout)
        if r.status_code != 200:
            raise HTTPException(502, r.text[:300])
        out = r.json()
        answers = {k: {"choice": a["choice"], "probabilities": a["probabilities"],
                       "confidence": a.get("confidence")}
                   for k, a in out["answers"].items() if a.get("type") == "choice"}
        return {"answers": answers,
                "routing": {"model": "llm", "checkpoint": out.get("model"), "name": args.name},
                "usage": {"input_tokens": (out.get("usage") or {}).get("input_tokens", 0), "cost": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
