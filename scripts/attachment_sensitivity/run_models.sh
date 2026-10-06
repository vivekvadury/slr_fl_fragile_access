#!/usr/bin/env bash
# Fit the seven transition models (both specifications, 199-success cluster
# bootstrap) for each attachment arm, writing ONLY to
#   outputs/attachment_sensitivity/<experiment>/models/<arm>/
# via TRANSITION_TABLE_DIR. Production outputs/tables/ is never written.
#
# Usage: bash scripts/attachment_sensitivity/run_models.sh <experiment_id> [max_parallel] [arm ...]
set -euo pipefail

EXP="${1:?experiment id required}"
MAX_PAR="${2:-6}"
shift $(( $# >= 2 ? 2 : $# ))
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(reference origin_1000 origin_500 facility_500 facility_add_uncapped facility_nearest_1000)
fi
RSCRIPT="${RSCRIPT:-/c/Program Files/R/R-4.6.1/bin/Rscript.exe}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="$REPO/data/processed/access/attachment_sensitivity/$EXP"
OUT_ROOT="$REPO/outputs/attachment_sensitivity/$EXP/models"
REPS="${AME_BOOT_REPS:-199}"
cd "$REPO"

run_one() {
  local arm="$1" spec="$2"
  local dir="$OUT_ROOT/$arm"
  local marker="$dir/_COMPLETE_${spec}.json"
  local data="$DATA_ROOT/$arm/block_group_analysis_dataset.csv"
  mkdir -p "$dir"
  if [ -f "$marker" ]; then echo "skip $arm/$spec (complete)"; return 0; fi
  [ -f "$data" ] || { echo "missing $data" >&2; return 2; }
  local start; start=$(date +%s)
  if TRANSITION_TABLE_DIR="$dir" BRIDGE_ARM=approach MODEL_SPEC="$spec" AME_BOOT_REPS="$REPS" \
      "$RSCRIPT" scripts/04_transition_models.R --data "$data" > "$dir/log_${spec}.txt" 2>&1; then
    printf '{"arm":"%s","spec":"%s","reps":%s,"seconds":%s,"completed":"%s"}\n' \
      "$arm" "$spec" "$REPS" "$(( $(date +%s) - start ))" "$(date -u +%FT%TZ)" > "$marker"
    echo "done $arm/$spec in $(( $(date +%s) - start )) s"
  else
    echo "FAILED $arm/$spec; see $dir/log_${spec}.txt" >&2
    return 1
  fi
}

pids=(); labels=(); status=0
for arm in "${ARMS[@]}"; do
  for spec in demographic_only with_physical; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
    run_one "$arm" "$spec" &
    pids+=("$!"); labels+=("$arm/$spec")
  done
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "job ${labels[$i]} failed" >&2; status=1; fi
done
exit $status
