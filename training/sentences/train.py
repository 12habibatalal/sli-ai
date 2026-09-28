"""Training stages: karsl (isolated-sign pre-training), isharah (sentences), distill (small from large).

    python -m training.sentences.train --stage isharah --model large --init karsl-large.pt
Checkpoints (best dev WER) and per-epoch logs go to $SLI_OUT.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from training.sentences import ctc
from training.sentences import data as D
from training.sentences import model as M
from training.sentences.vocab import encode

OUT = os.environ.get("SLI_OUT", ".")


def evaluate_split(model, feats, items, device):
    """Greedy-decoding WER over (key, ids) items, one sequence at a time."""
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for key, ids in items:
            x = torch.from_numpy(np.asarray(feats[key], np.float32))[None].to(device)
            lp, _ = model(x, torch.tensor([x.shape[1]], device=device))
            hyps.append(ctc.greedy(lp[0].float().cpu().numpy())[0])
            refs.append(ids)
    return ctc.wer(refs, hyps)


def stress_report(model, ckpt, feats, items, device, out_path):
    """Dev WER of the best checkpoint: as is, with a hand hidden in stretches, and 1.5x faster."""
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state"])
    model.to(device)
    report = {"plain": evaluate_split(model, feats, items, device)}
    for kind in ("hand", "fast", "noface"):
        rng = np.random.default_rng(0)
        hard = {k: D.stress(np.asarray(feats[k], np.float32), kind, rng) for k, _ in items}
        report[kind] = evaluate_split(model, hard, items, device)
    print("stress:", json.dumps(report), flush=True)
    json.dump(report, open(out_path, "w"))
    return report


def _seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def fit(model, feats, items, dev_feats, dev_items, epochs, lr, max_frames, device, augment=True,
        log=None, teacher=None, alpha=1.0, ckpt=None, meta=None, workers=0, aug="normal"):
    model.to(device)
    ds = D.SeqDataset(feats, items, train=augment, level=aug)
    lengths = [len(feats[k]) for k, _ in items]
    rng = np.random.default_rng(0)
    steps = epochs * len(D.bucket_batches(lengths, max_frames, rng))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    cuda = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=cuda)
    best, history = math.inf, []
    for ep in range(epochs):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        loader = torch.utils.data.DataLoader(ds, batch_sampler=D.bucket_batches(lengths, max_frames, rng),
                                             collate_fn=D.collate, num_workers=workers,
                                             worker_init_fn=_seed_worker if workers else None)
        for x, xl, y, yl in loader:
            x, xl = x.to(device, non_blocking=True), xl.to(device)
            with torch.autocast(device, enabled=cuda):
                lp, ol = model(x, xl)
            loss = ctc_loss(lp.float().transpose(0, 1), y, ol.cpu(), yl)
            if teacher is not None:  # frame-level distillation: same features, stride and fps
                with torch.no_grad(), torch.autocast(device, enabled=cuda):
                    tlp, _ = teacher(x, xl)
                valid = (torch.arange(lp.shape[1], device=device)[None] < ol[:, None]).float()
                kl = (tlp.float().exp() * (tlp.float() - lp.float())).sum(-1)
                loss = loss + alpha * (kl * valid).sum() / valid.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach())
            n += 1
        dev = evaluate_split(model, dev_feats, dev_items, device)
        history.append({"epoch": ep, "loss": total / max(n, 1), "dev_wer": dev, "sec": round(time.time() - t0)})
        print(json.dumps(history[-1]), flush=True)
        if dev < best:
            best = dev
            if ckpt:
                torch.save({"state": model.state_dict(), **(meta or {}), "dev_wer": dev, "epoch": ep}, ckpt)
        if log:
            json.dump(history, open(log, "w"))
    return best


def _path(data_dir, name):
    if data_dir:
        return os.path.join(data_dir, name)
    from training.sentences.kaggle_run import find_input
    return find_input(name)


def load_ish(data_dir=None):
    z = np.load(_path(data_dir, "ish_feats.npz"))
    feats = {k: z[k] for k in z.files}  # into RAM once (float16, a few GB)
    index = json.load(open(_path(data_dir, "ish_index.json"), encoding="utf-8"))
    vocab = json.load(open(_path(data_dir, "vocab.json"), encoding="utf-8"))
    return feats, index, vocab


def load_karsl(data_dir=None):
    z = np.load(_path(data_dir, "karsl_feats.npz"))
    return {k: z[k] for k in z.files}, json.load(open(_path(data_dir, "karsl_index.json")))


def _resolve(path):
    if path is None or os.path.exists(path):
        return path
    from training.sentences.kaggle_run import find_input
    return find_input(os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["karsl", "isharah", "distill", "stress"])
    ap.add_argument("--model", default="large")
    ap.add_argument("--init")
    ap.add_argument("--teacher")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--protocol", default="SI")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--aug", default="normal", choices=sorted(D.AUG))
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--keep-head", action="store_true", help="keep --init's CTC head (fine-tuning an Isharah checkpoint)")
    ap.add_argument("--data")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{a.stage}-{a.model}" + ("" if a.protocol == "SI" else f"-{a.protocol}")
    ckpt, log = os.path.join(OUT, name + ".pt"), os.path.join(OUT, name + ".log.json")
    lim = a.limit or None

    if a.stage == "karsl":
        kf, kidx = load_karsl(a.data)
        m = M.build(a.model, 503)  # 502 signs + blank
        tr = [(r["key"], [r["sign"]]) for r in kidx if r["split"] == "train"][:lim]
        dv = [(r["key"], [r["sign"]]) for r in kidx if r["split"] == "test"][::10][:lim]
        print(f"karsl: {len(tr)} train, {len(dv)} dev", flush=True)
        fit(m, kf, tr, kf, dv, a.epochs, a.lr, a.max_frames, device, log=log, ckpt=ckpt, workers=a.workers,
            meta={"config": a.model, "arch": M.CONFIGS[a.model], "n_classes": 503})
        return

    feats, index, vocab = load_ish(a.data)

    def split(part):
        rows = [r for r in index if r["splits"].get(a.protocol) == part]
        return [(r["id"], encode(vocab, r["gloss"])) for r in rows][:lim]

    C = len(vocab["glosses"]) + 1
    if a.stage == "stress":  # stress tests of an existing checkpoint (--init) on the dev split
        c = torch.load(_resolve(a.init), map_location="cpu")
        m = M.SignCTC(C, **(c.get("arch") or M.CONFIGS[c["config"]]))
        stress_report(m, _resolve(a.init), feats, split("dev"), device,
                      os.path.join(OUT, os.path.basename(a.init)[:-3] + ".stress.json"))
        return
    m = M.build(a.model, C, dropout=a.dropout)
    if a.init:
        state = torch.load(_resolve(a.init), map_location="cpu")["state"]
        if not a.keep_head:
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
        print("init:", m.load_state_dict(state, strict=False), flush=True)
    teacher = None
    if a.stage == "distill":
        t = torch.load(_resolve(a.teacher), map_location="cpu")
        teacher = M.SignCTC(C, **(t.get("arch") or M.CONFIGS[t["config"]]))
        teacher.load_state_dict(t["state"])
        teacher.to(device).eval()
    tr, dv = split("train"), split("dev")
    print(f"isharah {a.protocol}: {len(tr)} train, {len(dv)} dev, {C} classes", flush=True)
    fit(m, feats, tr, feats, dv, a.epochs, a.lr, a.max_frames, device, log=log, ckpt=ckpt, teacher=teacher,
        alpha=a.alpha, workers=a.workers, aug=a.aug,
        meta={"config": a.model, "arch": M.CONFIGS[a.model], "n_classes": C, "protocol": a.protocol,
              "aug": a.aug, "dropout": a.dropout})
    stress_report(m, ckpt, feats, dv, device, os.path.join(OUT, name + ".stress.json"))


if __name__ == "__main__":
    main()
