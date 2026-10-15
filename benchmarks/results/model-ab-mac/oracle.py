"""Parity oracle: sentence-transformers (upstream PyTorch weights, pinned
revision) vs onnxruntime on the exact ONNX files Axil pins, CLS and mean
pooling. Prints cosine agreement per text and the first 5 dims of the
ORT-CLS vector for comparison with the Rust embedder."""
import json, sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

TEXTS = [
    "The user graduated with a degree in Business Administration.",
    "We deployed a new Redis caching layer to the backend service.",
    "What programming language is best for systems work?",
    "fix(vector): pool gte-modernbert-base on [CLS], as upstream does. " * 60,
]

MODELS = {
    "granite": dict(repo="ibm-granite/granite-embedding-small-english-r2",
                    rev="2ab6fa8ea2d674564defd37171ae19079b864b33",
                    onnx=sys.argv[1] + "/onnx/model.onnx", tok=sys.argv[1] + "/tokenizer.json"),
    "gte": dict(repo="Alibaba-NLP/gte-modernbert-base",
                rev="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
                onnx=sys.argv[2] + "/model.onnx", tok=sys.argv[2] + "/tokenizer.json"),
}

def unit(v):
    return v / np.linalg.norm(v)

out = {}
for name, m in MODELS.items():
    st = SentenceTransformer(m["repo"], revision=m["rev"], device="cpu")
    ref = st.encode(TEXTS, normalize_embeddings=True, convert_to_numpy=True)
    tok = Tokenizer.from_file(m["tok"])
    tok.enable_truncation(8192)
    sess = ort.InferenceSession(m["onnx"], providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    rows = []
    for text, r in zip(TEXTS, ref):
        enc = tok.encode(text)
        feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        h = sess.run(None, feeds)[0][0]
        cls = unit(h[0])
        mean = unit(h.mean(axis=0))
        rows.append({"tokens": len(enc.ids), "st_vs_ort_cls": float(r @ cls),
                     "st_vs_ort_mean": float(r @ mean), "ort_cls_first5": [round(float(x), 5) for x in cls[:5]]})
    out[name] = {"inputs": names, "st_max_seq_length": st.max_seq_length, "rows": rows}
print(json.dumps(out, indent=1))
