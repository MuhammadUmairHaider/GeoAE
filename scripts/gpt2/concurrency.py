#!/usr/bin/env python3
"""Measure the unchanged GPT-2 recipe before sharing one GPU across layers.

The profile runs in disposable subprocesses and writes no checkpoints or
evaluation artifacts. It covers training, balanced fitting, token erasure and
LM inference; the most expensive evaluation stage sets the evaluation limit.
This is a memory admission estimate, not a guarantee against arbitrary OOMs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GIB = 2**30
HOST_HEADROOM_BYTES = 8 * GIB


def _read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _counters(text, *, kib=False):
    values = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = int(parts[1]) * (1024 if kib else 1)
            except ValueError:
                pass
    return values


def _mounts(proc_root):
    mounts = []
    for line in _read_text(proc_root / "self/mountinfo").splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if separator and len(fields) >= 5 and len(filesystem) >= 3:
            # mountinfo escapes whitespace and literal backslashes as octal.
            def unescape(value):
                for code, character in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
                    value = value.replace(code, character)
                return value
            mounts.append(dict(root=Path(unescape(fields[3])), path=Path(unescape(fields[4])),
                               filesystem=filesystem[0], options=filesystem[2].split(",")))
    return mounts


def reclaimable_regular_file(stat, version=2):
    """Credit clean regular-file LRU pages, while retaining all tmpfs/shmem.

    Kernel memory.stat's file includes shmem, but file LRU lists exclude it.
    Both active and inactive regular-file cache can be reclaimed. Bound that
    credit by the type counters, and subtract dirty/writeback and unevictable
    bytes conservatively; the counters need not form an exact partition.
    See https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files
    and https://docs.kernel.org/admin-guide/cgroup-v1/memory.html#stat-file.
    """
    prefix = "total_" if version == 1 and any(k.startswith("total_") for k in stat) else ""
    def value(name):
        return max(0, stat.get(prefix + name, 0))
    lru_keys = (prefix + "inactive_file", prefix + "active_file")
    file_key = prefix + ("file" if version == 2 else "cache")
    shmem_key = prefix + "shmem"
    lru = value("inactive_file") + value("active_file")
    bounds = []
    if any(key in stat for key in lru_keys):
        bounds.append(lru)
    if file_key in stat and shmem_key in stat:
        bounds.append(max(0, stat[file_key] - stat[shmem_key]))
    # Without either counter family, no cache credit is defensible.
    cache = min(bounds) if bounds else 0
    dirty = value("file_dirty" if version == 2 else "dirty")
    writeback = value("file_writeback" if version == 2 else "writeback")
    return max(0, cache - dirty - writeback - value("unevictable"))


def _cgroup_memory(proc_root, mounts):
    groups = []
    for line in _read_text(proc_root / "self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, name = parts
        version = 2 if hierarchy == "0" and not controllers else 1 if "memory" in controllers.split(",") else None
        if version is None:
            continue
        membership = Path(name)
        for mount in mounts:
            if mount["filesystem"] != ("cgroup2" if version == 2 else "cgroup"):
                continue
            if version == 1 and "memory" not in mount["options"]:
                continue
            try:
                relative = membership.relative_to(mount["root"])
            except ValueError:
                # A cgroup namespace can make membership relative to its root.
                relative = Path(str(membership).lstrip("/"))
            directory = mount["path"] / relative
            while directory == mount["path"] or mount["path"] in directory.parents:
                limit_file = directory / ("memory.max" if version == 2 else "memory.limit_in_bytes")
                usage_file = directory / ("memory.current" if version == 2 else "memory.usage_in_bytes")
                try:
                    current = int(_read_text(usage_file).strip())
                except ValueError:
                    current = None
                if current is not None:
                    text = _read_text(limit_file).strip()
                    limit = int(text) if text.isdigit() and int(text) < 2**60 else None
                    stat = _counters(_read_text(directory / "memory.stat"))
                    cache = min(current, reclaimable_regular_file(stat, version))
                    groups.append(dict(version=version, path=str(directory), limit_bytes=limit,
                                       current_bytes=current, reclaimable_regular_file_bytes=cache,
                                       irreducible_usage_bytes=max(0, current - cache),
                                       remaining_bytes=None if limit is None else max(0, limit - current + cache),
                                       memory_stat=stat))
                if directory == mount["path"]:
                    break
                directory = directory.parent
    return groups


def _staged_tmpfs_bytes(activations_dir, mounts):
    if activations_dir is None:
        return 0
    directory = Path(activations_dir).resolve()
    candidates = [m for m in mounts if directory == m["path"] or m["path"] in directory.parents]
    filesystem = max(candidates, key=lambda m: len(m["path"].parts))["filesystem"] if candidates else None
    if filesystem != "tmpfs" and not (filesystem is None and Path("/dev/shm") in directory.parents):
        return 0
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def host_memory_snapshot(activations_dir=None, *, environ=None, proc_root=Path("/proc"), staged_bytes=None):
    """Available bytes inside all ancestor limits, after clean file-cache reclaim.

    Slurm's node allocation is an additional bound, never the entire host's RAM.
    When job counters are inaccessible, the fallback charges staged tmpfs plus
    node-wide non-reclaimable non-shmem use. That may underestimate capacity
    when other jobs exist, but cannot substitute a 1 TiB node for a 450 GiB job.
    """
    environ = os.environ if environ is None else environ
    proc_root = Path(proc_root)
    mounts = _mounts(proc_root)
    groups = _cgroup_memory(proc_root, mounts)
    meminfo = _counters(_read_text(proc_root / "meminfo"), kib=True)
    staged = _staged_tmpfs_bytes(activations_dir, mounts) if staged_bytes is None else staged_bytes
    constraints = [dict(source=f"cgroup_v{g['version']}", path=g["path"], available_bytes=g["remaining_bytes"])
                   for g in groups if g["limit_bytes"] is not None]
    slurm_text = environ.get("SLURM_MEM_PER_NODE", "")
    slurm_limit = int(slurm_text) * 2**20 if slurm_text.isdigit() and int(slurm_text) > 0 else None
    fallback = None
    if slurm_limit is not None:
        job = environ.get("SLURM_JOB_ID", environ.get("SLURM_JOBID", ""))
        job_groups = [g for g in groups if any(p in (f"job_{job}", f"job_{job}.scope")
                                               for p in Path(g["path"]).parts)] if job else []
        # The shortest matching path is the job ancestor, including every step.
        if job_groups:
            group = min(job_groups, key=lambda g: len(Path(g["path"]).parts))
            usage = max(staged, group["irreducible_usage_bytes"])
            fallback = dict(source="slurm_job_cgroup", path=group["path"], irreducible_usage_bytes=usage)
        else:
            if "MemTotal" not in meminfo or "MemAvailable" not in meminfo:
                raise RuntimeError("Cannot measure host usage: no Slurm job cgroup or /proc/meminfo counters")
            # MemAvailable already includes reclaimable disk cache. Retain only
            # our known tmpfs instead of double charging global Shmem here.
            non_shmem = max(0, meminfo["MemTotal"] - meminfo["MemAvailable"] - meminfo.get("Shmem", 0))
            usage = staged + non_shmem
            fallback = dict(source="slurm_allocation_node_usage_fallback", staged_tmpfs_bytes=staged,
                            node_nonreclaimable_nonshmem_bytes=non_shmem, irreducible_usage_bytes=usage)
        constraints.append(dict(source=fallback["source"], available_bytes=max(0, slurm_limit - usage)))
    if not constraints:
        raise RuntimeError("No bounded memory cgroup or SLURM_MEM_PER_NODE; refusing to assume all host RAM is allocated")
    if "MemAvailable" in meminfo:
        constraints.append(dict(source="node_MemAvailable", available_bytes=meminfo["MemAvailable"]))
    limiting = min(constraints, key=lambda c: c["available_bytes"])
    return dict(available_bytes=limiting["available_bytes"], limiting_source=limiting,
                staged_tmpfs_bytes=staged, slurm_allocation_bytes=slurm_limit,
                cgroups=groups, fallback=fallback, constraints=constraints,
                node_MemAvailable_bytes=meminfo.get("MemAvailable"),
                cache_credit="min(active_file + inactive_file, file - shmem) - dirty - writeback - unevictable; clipped at zero")


def eval_host_reservation(n_rows=1_000_000, hidden_size=768, latent_dim=1536):
    """CQ holds raw baseline + latents; reserve an extra raw copy and native use."""
    if min(n_rows, hidden_size, latent_dim) <= 0:
        raise ValueError("Positive CQ row count and representation widths are required")
    raw, latent = n_rows * hidden_size * 4, n_rows * latent_dim * 4
    native = 4 * GIB  # Python/torch/sklearn, checkpoint/model, labels and metric scratch.
    total = math.ceil((2 * raw + latent + native) / GIB) * GIB
    return dict(reservation_bytes=total, n_rows=n_rows, hidden_size=hidden_size, latent_dim=latent_dim,
                raw_baseline_bytes=raw, latent_bytes=latent, extra_raw_copy_bytes=raw,
                native_model_metric_allowance_bytes=native, rounded_up_to_gib=True)


def cap_eval_host_memory(gpu_count, snapshot, reservation=None):
    reservation = eval_host_reservation() if reservation is None else reservation
    available = snapshot["available_bytes"]
    count = min(gpu_count, max(0, (available - HOST_HEADROOM_BYTES) // reservation["reservation_bytes"]))
    if count < 1:
        raise RuntimeError(f"Host memory cannot fit one evaluation worker: {available / GIB:.2f} GiB available, "
                           f"{reservation['reservation_bytes'] / GIB:.2f} GiB per worker plus "
                           f"{HOST_HEADROOM_BYTES / GIB:.2f} GiB shared headroom")
    return int(count), dict(**snapshot, worker_budget=reservation, headroom_bytes=HOST_HEADROOM_BYTES,
                            gpu_admitted_concurrency=gpu_count, admitted_concurrency=int(count))


def concurrency_limit(available_bytes, peak_bytes, *, requested=12,
                      process_overhead_bytes=512 * 2**20, safety_factor=1.15,
                      headroom_bytes=4 * GIB):
    """Reserve each process's measured peak, CUDA overhead and a safety margin."""
    if available_bytes <= 0 or peak_bytes <= 0 or not 1 <= requested <= 12:
        raise ValueError("Positive GPU memory and requested concurrency 1..12 are required")
    if process_overhead_bytes < 0 or headroom_bytes < 0 or safety_factor < 1:
        raise ValueError("GPU overhead/headroom must be nonnegative and safety factor >= 1")
    reservation = math.ceil(peak_bytes * safety_factor) + process_overhead_bytes
    count = min(requested, max(0, (available_bytes - headroom_bytes) // reservation))
    if count < 1:
        raise RuntimeError(f"One worker needs {reservation / GIB:.2f} GiB plus "
                           f"{headroom_bytes / GIB:.2f} GiB headroom; only "
                           f"{available_bytes / GIB:.2f} GiB is available")
    return int(count), reservation


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def process_gpu_bytes():
    """Driver memory includes context/library allocations outside torch's pool."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                 "--format=csv,noheader,nounits"], text=True,
                                capture_output=True, check=True, timeout=10)
        for line in result.stdout.splitlines():
            pid, memory = line.split(",", 1)
            if int(pid.strip()) == os.getpid():
                return int(memory.strip()) * 2**20
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0


def measure(args):
    # Lazy imports keep the planner itself CPU-testable.
    import numpy as np
    import torch
    from geoae.config import Config
    from geoae.paths import resolve_path
    from geoae.seeding import seed_everything
    from geoae.seeded_init import density_peaks_select
    from geoae.train_common import build_model, make_optimizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires the allocated CUDA device")
    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device("cuda", 0)
    # is_available() does not initialize the child process's CUDA allocator.
    torch.cuda.set_device(device)
    act_dir = resolve_path(args.activations_dir or cfg.data.activations_dir)
    raw = np.load(act_dir / f"layer_{cfg.data.target_layer}.npy", mmap_mode="r")
    if raw.shape[1] != cfg.model.hidden_size:
        raise ValueError("Activation width disagrees with the training config")
    samples = min(1_500_000, len(raw))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    def peaks(z, count, **kwargs):
        return density_peaks_select(z, count, seed=cfg.train.seed,
                                   density_power=cfg.train.density_power,
                                   min_sep_frac=cfg.train.peak_min_sep_frac,
                                   knn_k=cfg.train.density_knn,
                                   refine_k=cfg.train.peak_refine_k, **kwargs)[0]

    if args.measure == "train":
        from geoae import diagnostics
        from geoae.losses import total_loss
        model = build_model(cfg, device)
        opt = make_optimizer(model, cfg)
        norm = np.load(act_dir / f"norm_params_layer{cfg.data.target_layer}.npz")
        batch = cfg.data.batch_size
        if len(raw) < batch:
            raise ValueError("The profile requires one full scientific-recipe training batch")
        x = torch.from_numpy((raw[:batch].astype(np.float32) - norm["mean"]) / norm["std"]).to(device)
        tokens = None
        if model.has_token_bias:
            tokens = torch.from_numpy(np.load(act_dir / "rows_tok.npy", mmap_mode="r")[:batch].astype(np.int64)).to(device)
        with torch.no_grad():
            # Training retains this initialization pool until it exits.
            z_pool = model.encode(x, tokens)[:cfg.train.density_pool]
            model.init_centroids_from_class_means(peaks(z_pool, model.n_clusters))
        model.tau = cfg.loss.tau_end
        model.zipf_alpha = cfg.loss.zipf_alpha_end
        # Later phases retain more loss tensors than reconstruction-only warmup.
        for _ in range(3):
            out = model(x, tokens)
            losses = total_loss(x, out.x_hat, out.z, model.centroids.detach(), out.Q,
                                cfg.loss.lambda_cluster, cfg.loss.lambda_sep,
                                metric=model.metric, lambda_var=cfg.loss.lambda_var,
                                lambda_cov=cfg.loss.lambda_cov, lambda_unif=cfg.loss.lambda_unif,
                                sep_mode=cfg.loss.sep_mode, sep_margin=cfg.loss.sep_margin,
                                var_gamma=cfg.loss.var_gamma)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            model.post_step()
            model.update_centroids_ema(out.z.detach(), out.Q.detach())
        # Production validation/diagnostics run while the train output is alive.
        with torch.no_grad():
            model.eval()
            validation = model(x, tokens)
            validation_mse = (validation.x_hat - x).pow(2).mean().item()
            del validation
            model.train()
            diagnostics.compute_all(out.z.detach(), out.Q.detach(), model.centroids,
                                    0.0, validation_mse, 0.0, 0.0,
                                    [out.Q.detach().mean(0).cpu()])
            pool = out.z.detach()[:cfg.train.peak_reinit_pool]
            # Cover the largest possible reinitialization, independent of seed occupancy.
            peaks(pool, min(model.n_clusters, len(pool)))
    elif args.measure == "balanced":
        from geoae.interp.fit_balanced_kmeans import occupancy
        from geoae.losses import sinkhorn_log
        with torch.no_grad():
            X = torch.randn(samples, cfg.model.hidden_size, device=device)
            pool = X[torch.randperm(samples, device=device)[:cfg.train.density_pool]]
            C = peaks(pool, cfg.model.n_clusters).clone()
            del pool
            ema_size = torch.zeros(cfg.model.n_clusters, device=device)
            ema_sum = C.clone()
            perm = torch.randperm(samples, device=device)
            for start in range(0, 3 * 4096, 4096):
                b = X[perm[start:start + 4096]]
                cost = torch.cdist(b, C).pow(2)
                Q = sinkhorn_log(cost, 0.1, cfg.loss.sinkhorn_iters)
                hard = torch.nn.functional.one_hot(Q.argmax(1), cfg.model.n_clusters).to(b.dtype)
                cnt = hard.sum(0)
                ema_size.mul_(0.9).add_(cnt * 0.1)
                ema_sum.mul_(0.9).add_((hard.T @ b) * 0.1)
                visited = cnt > 0
                C[visited] = ema_sum[visited] / ema_size[visited, None].clamp_min(1e-6)
            peaks(b[:cfg.train.peak_reinit_pool], min(cfg.model.n_clusters, len(b)))
            occupancy(X[:1_000_000], C)
    elif args.measure == "erasure":
        from geoae.checkpoint import load_lm
        from geoae.interp.token_erasure import class_scatter, top_eigvecs
        with torch.no_grad():
            # Last-block diagnostics keep GPT-2 resident throughout the fit.
            lm = load_lm(cfg.extraction.model_name, device=device)
            rows = min(1_000_000, len(raw))
            Z = torch.randn(rows, cfg.model.hidden_size, device=device)
            # Production predicts labels before creating the centered copy.
            lm.get_output_embeddings()(Z[:8192].to(torch.bfloat16)).argmax(-1)
            # At min_count=30, this is the maximum possible class cardinality.
            key = np.arange(rows) % max(2, rows // 30)
            fitted = []
            for _ in range(2):
                S, classes, means, counts = class_scatter(Z, key, 30)
                fitted.append((classes, means, counts))
            top_eigvecs(S, min(512, cfg.model.hidden_size))
            Zc = Z - Z.mean(0)
            C_tot = torch.zeros_like(S)
            for start in range(0, len(Zc), 65536):
                blk = Zc[start:start + 65536].double()
                C_tot += blk.T @ blk
            top_eigvecs(C_tot / len(Zc), min(512, cfg.model.hidden_size))
    elif args.measure == "inference":
        from geoae.checkpoint import load_lm
        from geoae.hooks import SplicingHook
        lm = load_lm(cfg.extraction.model_name, device=device)
        ae = build_model(cfg, device)
        ae.eval()
        ae.centroids.normal_()
        ae.centroids_initialized.fill_(True)
        hook = SplicingHook(lm, cfg.data.target_layer)
        # MMLU defaults: 16 prompts, truncated to GPT-2's 1024-token context.
        ids = torch.zeros((16, 1024), dtype=torch.long, device=device)
        def splice(h):
            shape = h.shape
            x = h.flatten(0, 1).float()
            return ae(x, ids.flatten()).x_hat.reshape(shape).to(h.dtype)
        with torch.no_grad():
            base = lm(input_ids=ids, use_cache=False).logits
            hook.activate(splice)
            try:
                reconstructed = lm(input_ids=ids, use_cache=False).logits
                reconstructed[..., 0].sum().item()
            finally:
                hook.deactivate()
            del base, reconstructed
    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    driver = process_gpu_bytes()
    result = dict(stage=args.measure, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  process_overhead_bytes=max(512 * 2**20, driver - reserved),
                  synthetic_profile=True)
    atomic_json(args.out, result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["profile"], default="profile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--activations-dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--headroom-gib", type=float, default=4.0)
    parser.add_argument("--safety-factor", type=float, default=1.15)
    parser.add_argument("--measure", choices=["train", "balanced", "erasure", "inference"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.measure:
        return measure(args)
    import torch
    if not torch.cuda.is_available():
        parser.error("Profile must run on the allocated H100 after Gemma completes")
    free, total = torch.cuda.mem_get_info(0)
    profiles = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for stage in ("train", "balanced", "erasure", "inference"):
        destination = args.out.with_name(f"{args.out.stem}_{stage}.json")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", args.config,
                   "--out", str(destination), "--measure", stage]
        if args.activations_dir:
            command += ["--activations-dir", args.activations_dir]
        print(f"[gpu-profile] Measuring {stage} on {torch.cuda.get_device_name(0)}", flush=True)
        with destination.with_suffix(".log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        profiles[stage] = json.loads(destination.read_text())
        print(f"[gpu-profile] {stage}: {profiles[stage]['peak_reserved_bytes'] / GIB:.2f} GiB peak", flush=True)
    train = profiles["train"]
    evaluation = max((profiles[s] for s in ("balanced", "erasure", "inference")),
                     key=lambda item: item["peak_reserved_bytes"])
    options = dict(requested=args.max_concurrency, safety_factor=args.safety_factor,
                   headroom_bytes=math.ceil(args.headroom_gib * GIB))
    train_count, train_reservation = concurrency_limit(free, train["peak_reserved_bytes"],
                          process_overhead_bytes=train["process_overhead_bytes"], **options)
    eval_count, eval_reservation = concurrency_limit(free, evaluation["peak_reserved_bytes"],
                          process_overhead_bytes=max(p["process_overhead_bytes"] for p in profiles.values()), **options)
    from geoae.config import Config
    cfg = Config.from_yaml(args.config)
    host = host_memory_snapshot(args.activations_dir or cfg.data.activations_dir)
    budget = eval_host_reservation(hidden_size=cfg.model.hidden_size, latent_dim=cfg.model.latent_dim)
    eval_count, host_admission = cap_eval_host_memory(eval_count, host, budget)
    report = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  gpu_name=torch.cuda.get_device_name(0), available_bytes=free, total_bytes=total,
                  requested_concurrency=args.max_concurrency, train_concurrency=train_count,
                  eval_concurrency=eval_count, train_peak_bytes=train["peak_reserved_bytes"],
                  eval_peak_bytes=evaluation["peak_reserved_bytes"],
                  train_reservation_bytes=train_reservation, eval_reservation_bytes=eval_reservation,
                  host_memory=host_admission,
                  safety_factor=args.safety_factor, headroom_bytes=options["headroom_bytes"], profiles=profiles,
                  limitation="Synthetic tensor shapes cover known heavy stages; actual workloads may use more memory.")
    atomic_json(args.out, report)
    gc.collect()
    print(f"[gpu-profile] Selected {train_count}/12 training layers and {eval_count}/12 evaluation layers", flush=True)
    print(f"[host-admission] {host['available_bytes'] / GIB:.1f} GiB available after clean file-cache reclaim; "
          f"{budget['reservation_bytes'] / GIB:.0f} GiB per evaluation worker plus "
          f"{HOST_HEADROOM_BYTES / GIB:.0f} GiB shared headroom", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
