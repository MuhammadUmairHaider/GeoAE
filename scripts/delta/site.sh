# Per-account defaults for NCSA Delta. Sourced by scripts/delta/env.sh.
#
# This is the ONE file to edit when the allocation changes. Everything else
# (configs, eval modules, sbatch scripts, eval_out sheets) derives its paths
# from env.sh and carries no Delta-specific path.
#
# `accounts` lists the allocations; the directory name under /work/hdd,
# /work/nvme and /projects is the project part of the account name, so
# account bbyl-delta-gpu  ->  DELTA_PROJECT=bbyl.

# The allocation that holds the data, the venv and the GPU hours.
# 2026-09-26: moved off bgrb (115 GPU-h left, /projects/bgrb 84% full) to bbyl.
: "${DELTA_PROJECT:=bbyl}"
export DELTA_PROJECT

# Slurm accounts. GPU hours are the binding constraint on this project, so the
# sbatch scripts default to DELTA_ACCOUNT and every sheet can be pointed at a
# different allocation with `-A` or by exporting DELTA_ACCOUNT.
#   balances on 2026-09-26 (GPU-h):  bbyl 201  beto 383  bimc 288  bgrb 115
# CPU-only steps (the corpus build) should be charged to a *-delta-cpu account:
# bbyl has no CPU allocation, so DELTA_CPU_ACCOUNT points elsewhere.
: "${DELTA_ACCOUNT:=${DELTA_PROJECT}-delta-gpu}"
: "${DELTA_CPU_ACCOUNT:=bimc-delta-cpu}"
export DELTA_ACCOUNT DELTA_CPU_ACCOUNT

# WHERE THE BULK DATA LIVES — deliberately NOT under DELTA_PROJECT.
#
# Every allocation has three storage tiers (/projects, /work/hdd, /work/nvme), so with
# four allocations there are twelve directories and ~5 TB free between them. The 1 TB
# on each is an administrative project quota, not a disk: the underlying Lustre
# filesystem is petabytes, and `df` inside one of these trees reports the QUOTA as the
# filesystem size, which makes it look much smaller than it is.
#
# /work/hdd/bbyl is the crowded one — 822 GB of its 1 TB, EIGHT users sharing it,
# only ~178 GB free against a campaign that needs ~250 GB. /work/hdd/bimc is empty
# (1 TB, 0 used), so the dumps and checkpoints go there while the GPU hours are still
# charged to DELTA_ACCOUNT. Compute account and data location are independent: being
# a member of the group is what grants access, not what pays for the node.
#
# To move the data elsewhere, change this one line and re-run
# scripts/delta/link_data.sh — nothing else in the repo names a data path.
#   /work/hdd/bgrb  596 GB free   (the previous home, still holds the 2026-09 dumps)
#   /work/hdd/beto  223 GB free
#   /work/nvme/*    fast tier, ~500 GB each; better for a dump that is read hot
: "${DELTA_DATA:=/work/hdd/bimc/${USER}/GeoAE}"
export DELTA_DATA
