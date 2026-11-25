import re
import time
from collections import deque
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .rag import RAG, abstain
from .harness import CONFIG as HARNESS_CONFIG, classify_batch

app = FastAPI(title="GA5 RAG service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CollapseSlashes:
    """Treat //query (base URL submitted with a trailing slash) the same as /query."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and "//" in scope.get("path", ""):
            scope = dict(scope)
            scope["path"] = re.sub(r"/{2,}", "/", scope["path"])
            scope["raw_path"] = scope["path"].encode()
        await self.inner(scope, receive, send)


engine = RAG()  # loads both models once at startup
LOG: deque = deque(maxlen=300)  # last requests, readable at /debug/log


@lru_cache(maxsize=2048)
def _cached(question: str, top_k: int) -> dict:
    return engine.answer(question, top_k)


@app.get("/")
def root():
    return {"status": "ok", "usage": "POST /query {\"question\": str, \"top_k\": int}"}


@app.get("/debug/log")
def debug_log():
    return JSONResponse(list(LOG), headers={"Access-Control-Allow-Origin": "*"})


@app.post("/query")
async def query(request: Request):
    raw = None
    try:
        raw = await request.json()
    except Exception:
        pass
    body = raw if isinstance(raw, dict) else {}
    question = body.get("question") or body.get("query") or ""
    try:
        top_k = int(body.get("top_k") or 3)
    except Exception:
        top_k = 3
    try:
        from starlette.concurrency import run_in_threadpool

        out = dict(await run_in_threadpool(_cached, str(question), top_k))
    except Exception:
        out = abstain()
    LOG.append({"t": round(time.time(), 1), "body": raw, "resp": out})
    return JSONResponse(out, headers={"Access-Control-Allow-Origin": "*"})


HLOG: deque = deque(maxlen=16)  # last harness batches (request + our answer), readable at /debug/harness


@app.get("/harness")
def harness_config():
    return JSONResponse(HARNESS_CONFIG, headers={"Access-Control-Allow-Origin": "*"})


@app.post("/harness")
async def harness(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    out = classify_batch(body)
    HLOG.append({"t": round(time.time(), 1), "body": body, "resp": out})
    return JSONResponse(out, headers={"Access-Control-Allow-Origin": "*"})


@app.get("/debug/harness")
def debug_harness(i: int = -1):
    """Without ?i= : summary of logged batches. With ?i=N : full batch N."""
    items = list(HLOG)
    if i == -1:
        return JSONResponse([{"i": n, "t": x["t"], "batchId": x["body"].get("batchId"),
                              "cases": len(x["body"].get("cases") or [])} for n, x in enumerate(items)],
                            headers={"Access-Control-Allow-Origin": "*"})
    return JSONResponse(items[i] if 0 <= i < len(items) else {}, headers={"Access-Control-Allow-Origin": "*"})


app = CollapseSlashes(app)
