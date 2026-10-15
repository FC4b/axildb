#!/usr/bin/env python3
"""Fine-tune nli-deberta-v3-xsmall for U2 on harvested GitHub pairs; export ONNX.

U2's score is P(entailment | premise=LATER, hypothesis=U2_TEMPLATE.format(error))
(run_bakeoff.py). Training pushes that probability up on positives and down on
negatives with binary cross-entropy on the entailment probability; a negative may
put its remaining mass on contradiction or neutral, whichever the model prefers.
So the 3-way head and its label order stay as they are, and run_bakeoff.py
scores the tuned model with no special case (`--local`).

Model selection (the best evaluated checkpoint, the untuned model included)
uses the dev repos of the harvested pairs. The U2 test sets in sets/ are never
read here. `--drop-repos` leaves repos out of training and dev, for ablations.

On a GPU the run is not bit-reproducible across machines or driver versions
(some CUDA kernels are nondeterministic); compare seeds, not single runs.

Usage: python finetune_u2.py --out data/u2_finetune/run1 [--epochs 3] [--threads 16]
Requires requirements-finetune.txt.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_bakeoff import U2_TEMPLATE, auroc

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "data" / "u2_train" / "pairs.jsonl"
HARVEST_META = HERE / "data" / "u2_train" / "harvest-meta.json"

BASE = "cross-encoder/nli-deberta-v3-xsmall"
BASE_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
MAX_LEN = 128  # titles: the longest harvested pair is well under this (counted below)
MAX_ONNX_PROB_DIFF = 1e-4


def load_pairs(drop_repos):
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["repo"].lower() not in drop_repos]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]


def count_truncated(tok, rows):
    enc = tok([r["later"] for r in rows], [U2_TEMPLATE.format(r["error"]) for r in rows], truncation=False)
    return sum(1 for ids in enc["input_ids"] if len(ids) > MAX_LEN)


def encode(tok, rows):
    return tok([r["later"] for r in rows], [U2_TEMPLATE.format(r["error"]) for r in rows],
               truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


def entail_scores(model, tok, rows, ent, batch=64):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            enc = encode(tok, rows[i:i + batch]).to(dev)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            out.extend(torch.softmax(logits, dim=-1)[:, ent].tolist())
    return out


def bce_on_entailment(logits, labels, ent):
    """-log p_e for positives, -log(1 - p_e) for negatives."""
    lsm = torch.log_softmax(logits, dim=-1)
    log_pe = lsm[:, ent]
    others = torch.cat([lsm[:, :ent], lsm[:, ent + 1:]], dim=-1)
    log_not = torch.logsumexp(others, dim=-1)
    return -(labels * log_pe + (1 - labels) * log_not).mean()


def export_onnx(model, tok, out_dir, check_rows, ent):
    """Write model.onnx + tokenizer.json in the layout run_bakeoff.py --local reads,
    and check the ONNX probabilities against torch on dev pairs."""
    import onnxruntime as ort

    model.cpu().eval()
    enc = encode(tok, check_rows[:2])
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(out_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )
    tok_json = hf_hub_download(BASE, "tokenizer.json", revision=BASE_REVISION)
    (out_dir / "tokenizer.json").write_bytes(Path(tok_json).read_bytes())

    sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    worst = 0.0
    with torch.no_grad():
        for r in check_rows:
            e = encode(tok, [r])
            want = torch.softmax(model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).logits,
                                 dim=-1)[0].numpy()
            got = sess.run(None, {"input_ids": e["input_ids"].numpy(),
                                  "attention_mask": e["attention_mask"].numpy()})[0][0]
            got = np.exp(got - got.max())
            got = got / got.sum()
            worst = max(worst, float(np.abs(got - want).max()))
    if worst > MAX_ONNX_PROB_DIFF:
        raise SystemExit(f"ONNX export differs from torch by {worst:.2e} (> {MAX_ONNX_PROB_DIFF})")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir (keep it under data/, gitignored)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--drop-repos", default="", help="comma-separated owner/name list left out of train and dev")
    args = ap.parse_args()
    drop_repos = {r.strip().lower() for r in args.drop_repos.split(",") if r.strip()}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, dev = load_pairs(drop_repos)
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    truncated = {"train": count_truncated(tok, train), "dev": count_truncated(tok, dev)}
    model = AutoModelForSequenceClassification.from_pretrained(BASE, revision=BASE_REVISION).to(args.device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent = next(i for i, l in id2label.items() if l == "entailment")

    dev_y = [r["label"] for r in dev]
    base_auc = auroc(entail_scores(model, tok, dev, ent), dev_y)
    print(f"train {len(train)} ({sum(r['label'] for r in train)} pos), dev {len(dev)}; "
          f"dev AUROC before tuning {base_auc:.4f}", flush=True)

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = int(args.warmup * total)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    # The untuned weights are a candidate too: if no evaluation beats them,
    # they are what gets exported, and the meta says so.
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history, best = [], (base_auc, 0, initial)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for i in range(0, len(order), args.batch):
            model.train()
            batch = [train[j] for j in order[i:i + args.batch]]
            enc = encode(tok, batch).to(args.device)
            labels = torch.tensor([float(r["label"]) for r in batch], device=args.device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            loss = bce_on_entailment(logits, labels, ent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % eval_every == 0 or step == total:
                auc = auroc(entail_scores(model, tok, dev, ent), dev_y)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                                "loss": round(loss.item(), 4), "dev_auroc": round(auc, 4),
                                "elapsed_s": round(time.time() - t0)})
                print(json.dumps(history[-1]), flush=True)
                if auc > best[0]:
                    best = (auc, step, {k: v.detach().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[2])
    worst = export_onnx(model, tok, out_dir, dev[:50], ent)
    meta = {
        "base": BASE, "base_revision": BASE_REVISION, "objective": "bce on P(entailment)",
        "u2_template": U2_TEMPLATE, "max_len": MAX_LEN, "truncated_pairs": truncated, "args": vars(args),
        "pairs_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
        "harvest_meta": json.loads(HARVEST_META.read_text()) if HARVEST_META.exists() else None,
        "n_train": len(train), "n_train_pos": sum(r["label"] for r in train),
        "n_dev": len(dev), "dev_repos": sorted({r["repo"] for r in dev}),
        "dev_auroc_before": base_auc, "exported_dev_auroc": best[0], "exported_step": best[1],
        "history": history, "onnx_max_abs_prob_diff_vs_torch": worst,
        "train_seconds": round(time.time() - t0), "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "machine": f"{platform.machine()} {platform.platform()}",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "finetune-meta.json").write_text(json.dumps(meta, indent=2))
    print(f"best dev AUROC {best[0]:.4f} at step {best[1]}; ONNX vs torch max |dp| {worst:.2e}; "
          f"wrote {out_dir}")


if __name__ == "__main__":
    main()
