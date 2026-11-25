from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .rag import RAG, abstain

app = FastAPI(title="GA5 RAG service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = RAG()  # loads both models once at startup


class Query(BaseModel):
    question: str = ""
    top_k: int = 3


@lru_cache(maxsize=2048)
def _cached(question: str, top_k: int) -> dict:
    return engine.answer(question, top_k)


@app.get("/")
def root():
    return {"status": "ok", "usage": "POST /query {\"question\": str, \"top_k\": int}"}


@app.post("/query")
def query(body: Query):  # plain def -> runs in FastAPI's thread pool, so requests overlap
    try:
        out = dict(_cached(body.question or "", int(body.top_k or 3)))
    except Exception:
        out = abstain()
    return JSONResponse(out, headers={"Access-Control-Allow-Origin": "*"})
