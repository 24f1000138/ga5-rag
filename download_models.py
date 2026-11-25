"""Download the two ONNX models at Docker build time and verify their checksums."""
import hashlib, os, shutil
from huggingface_hub import hf_hub_download

OUT = os.environ.get("MODEL_DIR", "/app/models")
MODELS = {
    "emb": ("Xenova/all-MiniLM-L6-v2", "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e"),
    "ce": ("Xenova/ms-marco-MiniLM-L-6-v2", "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870"),
}
for sub, (repo, sha) in MODELS.items():
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    for f, dst in (("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")):
        shutil.copy(hf_hub_download(repo, f), os.path.join(d, dst))
    got = hashlib.sha256(open(os.path.join(d, "model.onnx"), "rb").read()).hexdigest()
    assert got == sha, f"checksum mismatch for {repo}: {got}"
    print("ok", repo)
