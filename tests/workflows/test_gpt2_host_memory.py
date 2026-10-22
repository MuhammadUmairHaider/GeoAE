"""Host admission honors the job allocation while retaining staged tmpfs."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpt2_host_admission_test", ROOT / "experiments/gpt2/concurrency.py")
concurrency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concurrency)
GIB = concurrency.GIB


def proc_fixture(tmp_path, membership, *, version=2, mount_root="/"):
    proc, mount = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    fs = "cgroup2" if version == 2 else "cgroup"
    options = "rw" if version == 2 else "rw,memory"
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {mount_root} {mount} rw - {fs} {fs} {options}\n")
    (proc / "self/cgroup").write_text(f"0::{membership}\n" if version == 2 else f"3:memory:{membership}\n")
    (proc / "meminfo").write_text(f"MemTotal: {1024 * GIB // 1024} kB\nMemAvailable: {600 * GIB // 1024} kB\nShmem: {344 * GIB // 1024} kB\n")
    return proc, mount


def write_group(directory, limit, current, stat, version=2):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ("memory.max" if version == 2 else "memory.limit_in_bytes")).write_text(str(limit))
    (directory / ("memory.current" if version == 2 else "memory.usage_in_bytes")).write_text(str(current))
    (directory / "memory.stat").write_text("".join(f"{k} {v}\n" for k, v in stat.items()))


def test_job_ancestor_wins_over_empty_step_and_unbounded_leaf(tmp_path):
    proc, mount = proc_fixture(tmp_path, "/system.slice/job_284386/step_batch/user/task_0")
    job = mount / "system.slice/job_284386"
    # Staged shmem remains charged; active GPFS cache can be reclaimed.
    write_group(job, 450 * GIB, 448 * GIB, dict(file=440 * GIB, shmem=344 * GIB,
                active_file=96 * GIB, inactive_file=0, file_dirty=2 * GIB, file_writeback=GIB))
    write_group(job / "step_batch/user", 450 * GIB, 9 * 2**20, {})
    write_group(job / "step_batch/user/task_0", "max", 2**20, {})
    report = concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB,
                environ={"SLURM_JOB_ID": "284386", "SLURM_MEM_PER_NODE": "460800"})
    assert report["available_bytes"] == 95 * GIB  # 450 - 448 + (96 - 2 - 1)
    group = next(g for g in report["cgroups"] if g["path"] == str(job))
    assert group["current_bytes"] == 448 * GIB
    assert group["reclaimable_regular_file_bytes"] == 93 * GIB
    count, provenance = concurrency.cap_eval_host_memory(12, report)
    assert count == 5
    assert provenance["gpu_admitted_concurrency"] == 12


def test_tighter_parent_limit_is_obeyed_even_when_leaf_has_space(tmp_path):
    proc, mount = proc_fixture(tmp_path, "/job_7/step/task")
    write_group(mount / "job_7", 450 * GIB, 390 * GIB, {})
    write_group(mount / "job_7/step", 450 * GIB, GIB, {})
    result = concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB, environ={})
    assert result["available_bytes"] == 60 * GIB
    assert result["limiting_source"]["path"] == str(mount / "job_7")


def test_shmem_is_never_credited_as_regular_file_cache():
    stat = dict(file=400 * GIB, shmem=344 * GIB, active_file=100 * GIB, inactive_file=10 * GIB,
                file_dirty=2 * GIB, file_writeback=GIB, unevictable=GIB)
    assert concurrency.reclaimable_regular_file(stat) == 52 * GIB
    assert concurrency.reclaimable_regular_file(dict(file=344 * GIB, shmem=344 * GIB)) == 0
    assert concurrency.reclaimable_regular_file(dict(file=344 * GIB)) == 0


def test_actual_argo_active_gpfs_cache_is_reclaimable():
    stat = dict(anon=3029123072, file=153695698944, shmem=21241856,
                inactive_file=8192, active_file=153674424320, file_dirty=501501952, file_writeback=0)
    assert concurrency.reclaimable_regular_file(stat) == 153172930560


def test_cgroup_v1_uses_hierarchical_cache_totals(tmp_path):
    proc, mount = proc_fixture(tmp_path, "/job_3/task", version=1)
    write_group(mount / "job_3", 450 * GIB, 440 * GIB,
                dict(total_cache=420 * GIB, total_shmem=344 * GIB, total_active_file=76 * GIB,
                     total_inactive_file=0, total_dirty=GIB, total_writeback=GIB), version=1)
    write_group(mount / "job_3/task", 9223372036854771712, GIB, {}, version=1)
    report = concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB, environ={})
    assert report["available_bytes"] == 84 * GIB
    assert report["limiting_source"]["source"] == "cgroup_v1"


def test_namespaced_cgroup_mount_maps_membership_to_mount_root(tmp_path):
    proc, mount = proc_fixture(tmp_path, "/job_3/task", mount_root="/job_3")
    write_group(mount, 450 * GIB, 380 * GIB, {})
    write_group(mount / "task", "max", GIB, {})
    assert concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB, environ={})["available_bytes"] == 70 * GIB


def test_fallback_uses_slurm_allocation_not_one_tb_node(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    # Node non-reclaimable use=400 GiB, including 344 GiB shmem.
    (proc / "meminfo").write_text(f"MemTotal: {1024 * GIB // 1024} kB\nMemAvailable: {624 * GIB // 1024} kB\nShmem: {344 * GIB // 1024} kB\n")
    report = concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB,
                                            environ={"SLURM_MEM_PER_NODE": "460800"})
    assert report["slurm_allocation_bytes"] == 450 * GIB
    assert report["available_bytes"] == 50 * GIB
    assert report["fallback"]["node_nonreclaimable_nonshmem_bytes"] == 56 * GIB
    assert concurrency.cap_eval_host_memory(12, report)[0] == 2


def test_system_memory_pressure_also_limits_workers(tmp_path):
    proc, mount = proc_fixture(tmp_path, "/job_3")
    write_group(mount / "job_3", 450 * GIB, 344 * GIB, {})
    (proc / "meminfo").write_text(f"MemTotal: {1024 * GIB // 1024} kB\nMemAvailable: {30 * GIB // 1024} kB\n")
    report = concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB, environ={})
    assert report["available_bytes"] == 30 * GIB
    assert concurrency.cap_eval_host_memory(12, report)[0] == 1


def test_no_allocation_evidence_fails_instead_of_guessing_host_ram(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(f"MemTotal: {1024 * GIB // 1024} kB\nMemAvailable: {600 * GIB // 1024} kB\n")
    with pytest.raises(RuntimeError, match="refusing to assume"):
        concurrency.host_memory_snapshot(proc_root=proc, staged_bytes=344 * GIB, environ={})


def test_host_budget_rejects_one_worker_that_does_not_fit():
    budget = concurrency.eval_host_reservation()
    assert budget["reservation_bytes"] == 16 * GIB
    assert budget["raw_baseline_bytes"] == 1_000_000 * 768 * 4
    assert budget["latent_bytes"] == 1_000_000 * 1536 * 4
    with pytest.raises(RuntimeError, match="cannot fit one"):
        concurrency.cap_eval_host_memory(12, dict(available_bytes=23 * GIB))


def test_host_admission_preserves_lower_gpu_limit():
    count, report = concurrency.cap_eval_host_memory(3, dict(available_bytes=105 * GIB))
    assert count == 3
    assert report["admitted_concurrency"] == 3
