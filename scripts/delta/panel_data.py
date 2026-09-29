#!/usr/bin/env python3
"""
Snapshot of the round-2 bypass-AE width series for the status panel, as JSON.

    python3 scripts/delta/panel_data.py > /path/to/panel.json

Stdlib only (runs under the system python3, no venv). Reads the per-config
training logs, the checkpoint dirs, squeue/sacct, `accounts` and `quota`.
Never submits, cancels or modifies anything.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (key, latent width, K). key is what jobs, steps (head-<key>) and the page use.
RUNS = [("d768", 768, 2000), ("d3072", 3072, 2000), ("d6144", 6144, 2000),
        ("d12288", 12288, 2000), ("d6144k4000", 6144, 4000),
        ("d12288k4000", 12288, 4000)]
N_EPOCHS = 50
STEPS_PER_EPOCH = 289
CFG = "llama3.2-3b_l27_k{k}_bnh_b32k_lam1_d{w}_dpc_sampled_tokbias"
CKPT = "checkpoints/llama3.2-3B/layer27/k{k}_bnh_b32k_lam1_d{w}_dpc_sampled_tokbias"
SINCE = "2026-09-26T16:00"

STEP_RE = re.compile(r"^\s+step\s+(\d+) \| .*?val_mse ([\d.]+) \| fve ([-\d.]+) \| eff_K (\d+)/(\d+) \| dying (\d+)")
HDR_RE = re.compile(r"^\[train\] Epoch (\d+)/(\d+) \|")
DONE_RE = re.compile(r"^\[train\] Epoch (\d+) done in ([\d.]+) min")
SAVE_RE = re.compile(r"^\[train\] Saved checkpoint: (step_\d+\.pt)\s+val_mse=([\d.]+)")
BEST_RE = re.compile(r"New best val_mse=([\d.]+)")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def parse_log(path: Path) -> dict:
    hist: dict[int, dict] = {}
    cur = None
    last_step = None
    best = None
    resumed = 0
    if not path.exists():
        return {"history": [], "current_epoch": None, "last_step": None, "best_val": None, "resumes": 0}
    for line in path.read_text(errors="replace").splitlines():
        if m := HDR_RE.match(line):
            cur = int(m.group(1))
            # a header for epoch N (fresh start or resume) invalidates N and later
            for e in [e for e in hist if e >= cur]:
                del hist[e]
            continue
        if line.startswith("[resume] resuming at epoch"):
            resumed += 1
        if m := STEP_RE.match(line):
            last_step = {"step": int(m.group(1)), "val_mse": float(m.group(2)), "fve": float(m.group(3)),
                         "eff_k": int(m.group(4)), "dying": int(m.group(6)), "epoch": cur}
            if cur is not None:
                hist.setdefault(cur, {"epoch": cur}).update(
                    fve=last_step["fve"], dying=last_step["dying"], eff_k=last_step["eff_k"])
            continue
        if (m := SAVE_RE.match(line)) and cur is not None:
            hist.setdefault(cur, {"epoch": cur}).update(ckpt=m.group(1), val_mse=float(m.group(2)))
            continue
        if m := BEST_RE.search(line):
            best = float(m.group(1))
        if m := DONE_RE.match(line):
            e = int(m.group(1))
            hist.setdefault(e, {"epoch": e}).update(minutes=float(m.group(2)), done=True)
    done = [hist[e] for e in sorted(hist) if hist[e].get("done")]
    return {"history": done, "current_epoch": cur, "last_step": last_step, "best_val": best, "resumes": resumed}


def slurm_time_to_h(t: str) -> float | None:
    if not t or t in ("N/A", "UNLIMITED", "INVALID"):
        return None
    d = 0
    if "-" in t:
        d, t = t.split("-", 1)
        d = int(d)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return d * 24 + h + m / 60 + s / 3600


def key_of(text: str) -> str | None:
    m = re.search(r"_k(\d+)_bnh_b32k_lam1_d(\d+)_dpc_sampled_tokbias", text or "")
    if not m:
        return None
    k, w = int(m.group(1)), int(m.group(2))
    return f"d{w}" + ("" if k == 2000 else f"k{k}")


def jobs() -> tuple[list, list, list]:
    # current queue
    q = []
    for line in sh(f"squeue -u $USER -h -o '%i|%a|%P|%T|%M|%l|%N|%r|%E|%j|%S'").splitlines():
        f = line.split("|")
        if len(f) < 11:
            continue
        q.append(dict(id=f[0], account=f[1], partition=f[2], state=f[3], elapsed=f[4], limit=f[5],
                      node=f[6], reason=f[7], dependency=f[8], name=f[9], start=f[10]))
    # job steps (head starts run as steps inside another job's allocation)
    steps = []
    for line in sh("squeue -s -u $USER -h -o '%i|%j|%M|%N'").splitlines():
        f = line.split("|")
        if len(f) >= 4 and f[1].startswith("head-d"):
            steps.append(dict(id=f[0], name=f[1], elapsed=f[2], node=f[3], key=f[1][len("head-"):]))
    # campaign history with submit lines (width mapping)
    hist = []
    out = sh(f"sacct -u $USER -S {SINCE} -X -n -P --format=JobID,JobName,Account,Partition,State,Start,End,Elapsed,NodeList,ExitCode,SubmitLine")
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 11 or not f[1].startswith("geoae"):
            continue
        hist.append(dict(id=f[0], name=f[1], account=f[2], partition=f[3], state=f[4].split()[0],
                         start=f[5], end=f[6], elapsed=f[7], node=f[8], exit=f[9],
                         key=key_of("|".join(f[10:]))))
    wmap = {h["id"]: h["key"] for h in hist}
    missing = [j["id"] for j in q if j["id"] not in wmap]
    if missing:  # pending-on-dependency jobs are not "eligible" yet, so -S skips them
        for line in sh(f"sacct -j {','.join(missing)} -X -n -P --format=JobID,SubmitLine").splitlines():
            f = line.split("|", 1)
            if len(f) == 2:
                wmap[f[0]] = key_of(f[1])
    for j in q:
        j["key"] = wmap.get(j["id"])
    return q, steps, hist


def accounts() -> list:
    rows = []
    for line in sh("accounts").splitlines():
        m = re.match(r"^(\S+-delta-(?:gpu|cpu))\s+(\d+)\s+(\d+)", line)
        if m:
            rows.append(dict(account=m.group(1), balance=int(m.group(2)), deposited=int(m.group(3))))
    return rows


def storage() -> list:
    rows = []
    want = {"/work/hdd/bimc": "dumps + checkpoints", "/work/hdd/bbyl": "shared, crowded",
            "/work/nvme/bbyl": "venv", "/projects/bbyl": "code"}
    for line in sh("quota").splitlines():
        f = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(f) >= 3 and f[0] in want:
            rows.append(dict(path=f[0], role=want[f[0]], used=f[1], quota=f[2]))
    return rows


def size_gb(p: Path) -> float:
    return round(sum(x.stat().st_size for x in p.glob("step_*.pt")) / 1e9, 1) if p.exists() else 0.0


def main() -> None:
    now = datetime.now()
    q, steps, hist = jobs()
    runs = []
    for key, w, k in RUNS:
        lg = parse_log(ROOT / "logs" / f"{CFG.format(w=w, k=k)}.log")
        h = lg["history"]
        ck = ROOT / CKPT.format(w=w, k=k)
        ckpts = sorted(ck.glob("step_*.pt")) if ck.exists() else []
        done = h[-1]["epoch"] if h else 0
        mins = [x["minutes"] for x in h[-3:] if "minutes" in x]
        mpe = round(statistics.median(mins), 1) if mins else None
        step = next((s for s in steps if s["key"] == key), None)
        run_job = next((j for j in q if j["key"] == key and j["state"] == "RUNNING"), None)
        pend_job = next((j for j in q if j["key"] == key and j["state"] == "PENDING"), None)
        if done >= N_EPOCHS:
            state, where = "done", "finished"
        elif step:
            state, where = "head start", f"{step['node']} · step {step['id']}"
        elif run_job:
            state, where = "running", f"{run_job['node']} · job {run_job['id']}"
        elif pend_job:
            state, where = "queued", f"job {pend_job['id']} · {pend_job['reason']}"
        else:
            state, where = "stopped", "not in queue"
        left = N_EPOCHS - done
        eta = None
        if state == "running" and mpe:
            eta = (now + timedelta(minutes=left * mpe)).strftime("%a %H:%M")
        last = h[-1] if h else {}
        runs.append(dict(
            key=key, width=w, k=k, label=f"d{w}" + ("" if k == 2000 else f" K{k}"),
            short=f"d{w}" if k == 2000 else f"K{k}", config=CFG.format(w=w, k=k), state=state, where=where,
            job=(run_job or pend_job or {}).get("id"), queued_job=(pend_job or {}).get("id"),
            queued_account=(pend_job or {}).get("account"), queued_partition=(pend_job or {}).get("partition"),
            epochs_done=done, n_epochs=N_EPOCHS, current_epoch=lg["current_epoch"],
            min_per_epoch=mpe, hours_left=round(left * mpe / 60, 1) if mpe else None, eta=eta,
            val_mse=last.get("val_mse"), fve=last.get("fve"), dying=last.get("dying"), eff_k=last.get("eff_k"),
            best_val=lg["best_val"], resumes=lg["resumes"], live=lg["last_step"],
            ckpts=len(ckpts), latest_ckpt=ckpts[-1].name if ckpts else None, ckpt_gb=size_gb(ck),
            final_ckpt=f"step_{N_EPOCHS * STEPS_PER_EPOCH:07d}.pt",
            host_job=(step["id"].split(".")[0] if step else None),
            history=[{k: x.get(k) for k in ("epoch", "val_mse", "fve", "dying", "minutes")} for x in h],
        ))
    # honest usage from the latest checkpoint of each run (needs torch -> venv, cached)
    cache = ROOT / "logs" / "panel_usage.json"
    dirs = [str(ROOT / CKPT.format(w=w, k=k)) for _, w, k in RUNS]
    sh(f"cd {ROOT} && scripts/delta/py scripts/delta/panel_usage.py {cache} " + " ".join(dirs))
    usage = json.loads(cache.read_text()) if cache.exists() else {}
    for r, d in zip(runs, dirs):
        r["usage"] = {k: v for k, v in usage.get(d, {}).items() if k != "key"} or None
    for r in runs:
        if r["state"] != "head start" or not r["min_per_epoch"]:
            continue
        host = next((x for x in runs if x["job"] == r["host_job"] and x["state"] == "running"), None)
        if host and host["hours_left"] is not None:
            reach = min(N_EPOCHS, r["epochs_done"] + int(host["hours_left"] * 60 // r["min_per_epoch"]))
            r["stops_at"] = host["eta"]
            r["stops_epoch"] = reach
            r["hours_after_restart"] = round((N_EPOCHS - reach) * r["min_per_epoch"] / 60, 1)
    meta = ROOT / "activations_sampled_10M" / "meta.json"
    dump = {}
    if meta.exists():
        m = json.loads(meta.read_text())
        dump = dict(rows=m["n_tokens"], docs=m["n_docs"], mix=m["source_share"],
                    median_norm=round(m["outliers"]["median_norm"], 2),
                    over10x=m["outliers"]["frac_over"].get(">10x"),
                    positions=m["position_hist"], hours=round(m["elapsed_seconds"] / 3600, 2),
                    doc_len_median=m["doc_length"]["median"])
    tb = {}
    tbl = ROOT / "logs" / "token_bias_activations_sampled_10M.log"
    if tbl.exists():
        t = tbl.read_text(errors="replace")
        if m := re.search(r"([\d,]+) tokens with >= 10 rows cover ([\d.]+)% of train rows", t):
            tb["tokens"], tb["train_coverage"] = m.group(1), float(m.group(2))
        if m := re.search(r"val rows: coverage ([\d.]+)%, variance removed ([\d.]+)", t):
            tb["val_coverage"], tb["var_removed"] = float(m.group(1)), float(m.group(2))
        if m := re.search(r"\((\d+) MB fp16\)", t):
            tb["mb"] = int(m.group(1))
    json.dump(dict(generated=now.strftime("%Y-%m-%d %H:%M"), generated_label=now.strftime("%a %d %b, %H:%M CDT"),
                   runs=runs, queue=q, steps=steps, history=hist, accounts=accounts(), storage=storage(),
                   dump=dump, token_bias=tb,
                   wandb="https://wandb.ai/multifacetednlp/geosep-general-e2e"),
              fp=__import__("sys").stdout, indent=1)


if __name__ == "__main__":
    main()
