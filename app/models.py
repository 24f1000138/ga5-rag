"""Small ONNX wrappers for the two models, so the image needs no PyTorch.

- Embedder: all-MiniLM-L6-v2 (bi-encoder). Turns text into a 384-dim vector.
- Reranker: ms-marco-MiniLM-L-6-v2 (cross-encoder). Reads question + passage
  together and outputs one relevance logit. Slower but much sharper, which is
  what separates near-identical "confounder" passages.
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False  # keeps memory low on small free instances
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class _Base:
    def __init__(self, sub: str, max_len: int):
        self.tok = Tokenizer.from_file(str(MODEL_DIR / sub / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.sess = _session(MODEL_DIR / sub / "model.onnx")
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _feed(self, encs):
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        return feed, mask


class Embedder(_Base):
    def __init__(self):
        super().__init__("emb", 256)

    def encode(self, texts: list[str]) -> np.ndarray:
        feed, mask = self._feed(self.tok.encode_batch(texts))
        hidden = self.sess.run(None, feed)[0]  # (n, seq, 384)
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)


class Reranker(_Base):
    def __init__(self):
        super().__init__("ce", 512)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        feed, _ = self._feed(self.tok.encode_batch([(query, d) for d in docs]))
        return self.sess.run(None, feed)[0][:, 0]  # raw logits
