#!/bin/bash
# South Florida SLR access-classification Slurm template.
#
# Before submitting:
#   1. Replace every TODO resource and module value below.
#   2. Run `mkdir -p logs` from the repository root. Slurm opens log files
#      before the job script can create the directory.
#
# NetworkX graph algorithms used here are single-threaded. Memory and walltime
# matter much more than CPU count; extra cores will not accelerate 2ecc work.

#SBATCH --job-name=slr_access
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=TODO_HH:MM:SS
#SBATCH --mem=TODO_GB
#SBATCH --cpus-per-task=TODO_CORES
#SBATCH --output=logs/access_%x_%j.out
#SBATCH --error=logs/access_%x_%j.err

set -euo pipefail

# TODO: replace with the cluster's Python/GDAL/GEOS module stack.
module purge
module load TODO_PYTHON_MODULE
module load TODO_GEOSPATIAL_MODULE

CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME, e.g. corrected_canonical}"
BRIDGE_RULE="${BRIDGE_RULE:-approach}"
SCENARIOS="${SCENARIOS:-0,1,2,3,4,5,6}"
RUN_PARENT="${RUN_PARENT:-data/processed/access/edited/cluster_runs}"

# Keep large parquet graph caches on node/job scratch, never in $HOME.
# For cache reuse across separate jobs, replace SLURM_TMPDIR with a persistent
# project-scratch path approved for your cluster.
SCRATCH_ROOT="${SLURM_TMPDIR:?SLURM_TMPDIR is not set}"
CACHE_DIR="${CACHE_DIR:-${SCRATCH_ROOT}/slr_access_graph_cache}"
mkdir -p "${CACHE_DIR}" "${RUN_PARENT}"

python scripts/02_access_flags.py \
  --run-dir "${RUN_PARENT}" \
  --cache-dir "${CACHE_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --bridge-rule "${BRIDGE_RULE}" \
  --scenarios "${SCENARIOS}" \
  --resume
