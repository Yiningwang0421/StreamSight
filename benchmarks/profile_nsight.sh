#!/usr/bin/env bash
# Profile the custom kernels with NVIDIA Nsight Compute.
# Ordinary training, tests, and benchmarks do not require this script.
#
# Usage, from streamsight-cuda/, after `python setup.py build_ext --inplace`:
#   bash benchmarks/profile_nsight.sh
#
# What to read in the report
#   Launch Statistics
#     grid size, block size, registers per thread, static shared memory
#   Occupancy
#     achieved occupancy versus the theoretical limit for this register/shared-memory mix
#   Memory Workload Analysis
#     DRAM throughput, sector utilization, and whether global loads are coalesced
#   Warp State Statistics
#     how much issue time is stalled on long-scoreboard (memory) versus math
#   Source Counters, if the build kept line info
#     which loop in the KV scan or the K-tile loop dominates
#
# Why the numbers move
#   Decode latency grows about linearly with valid_length because one warp scans
#   the cache. Batch * heads is the only grid dimension, so a small batch leaves
#   SMs idle. That is a property of this one-warp-per-head launch, not of FlashAttention-2.
#   Implicit GEMM K equals Cin * 9, so more input channels mean more K tiles and
#   more scattered activation gathers. BM/BN/BK are fixed at 16/16/8; a different
#   tile would trade occupancy against reused bytes, which this script does not sweep.
#   A large GPU with few warps will show low achieved occupancy even if each warp
#   is doing the right memory pattern.

set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu is not on PATH. Install NVIDIA Nsight Compute and rerun this script." >&2
  echo "The rest of the project does not need it." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results

ncu --set full -o results/ncu_decode --target-processes all \
  python benchmarks/profile_once.py --kernel decode --batch 4 --context 256

ncu --set full -o results/ncu_conv --target-processes all \
  python benchmarks/profile_once.py --kernel conv --batch 2

echo "Wrote results/ncu_decode.ncu-rep and results/ncu_conv.ncu-rep"
echo "Open them with the Nsight Compute UI, or print a summary with:"
echo "  ncu --import results/ncu_decode.ncu-rep"
