---
title: GA5 RAG
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GA5 RAG service

`POST /query` with `{"question": "...", "top_k": 3}` returns
`{"answer", "sources", "scores", "confidence"}` over the 50-passage knowledge base in `kb.json`.

- Retrieval: all-MiniLM-L6-v2 embeddings + ms-marco-MiniLM-L-6-v2 cross-encoder reranker (ONNX, CPU).
- Subject aliases pin the right passage when two passages look alike (Raft vs Multi-Paxos, SGD vs AdamW, ...).
- Answers are short values copied from the passage text (port, ms, ratio, count, bits, hash).
- Abstains (null answer, empty sources, confidence 0.1) when the KB does not contain the answer.
- Instruction-like text inside the question is stripped before retrieval.
