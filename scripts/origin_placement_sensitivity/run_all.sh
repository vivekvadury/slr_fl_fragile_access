#!/usr/bin/env bash
# Origin-placement Monte Carlo sensitivity: full pipeline (exploratory).
#
# Stages (each resumable; outputs under outputs/origin_placement_sensitivity/):
#   01  geometry, 50 draws/block (land polygon), production LCC snaps      ~3 min
#   02  per-scenario dry graph + node-state table + production checks      5-22 min/scenario
#       (6 ft: 73 min, dominated by the unmodified production-function pilot check)
#   03  classify pilot (10 draws) -> STOP if any validation count != 0
#   03  classify full (50 draws)
#   04  summaries, figures, block-group datasets (pilot and full)
#   05  demographic-only transition models on every replication (R)
#   06  depth mechanism diagnostic; 07 model comparison; 08 run manifest
#
# Usage (repo root):  bash scripts/origin_placement_sensitivity/run_all.sh
# Env: PY (python with geopandas/networkx), RSCRIPT. Peak memory ~6 GiB (stage 02).
set -euo pipefail
PY="${PY:-/c/Users/Vivek/miniforge3/envs/research-geo/python.exe}"
RSCRIPT="${RSCRIPT:-/c/Program Files/R/R-4.6.1/bin/Rscript.exe}"
export MKL_THREADING_LAYER=SEQUENTIAL PYTHONIOENCODING=utf-8
D=scripts/origin_placement_sensitivity
L=outputs/origin_placement_sensitivity/logs
W=outputs/origin_placement_sensitivity/work
mkdir -p "$L"

[ -f "$W/prepare_manifest.json" ] || "$PY" "$D/01_prepare_draws.py" > "$L/01_prepare_draws.log" 2>&1
for s in 0 1 2 3 4 5 6; do
  [ -f "$W/node_states_slr_${s}ft_checks.json" ] || "$PY" "$D/02_build_node_states.py" --scenario "$s" > "$L/02_node_states_${s}ft.log" 2>&1
done
"$PY" "$D/03_classify_draws.py" --subset pilot --n-draws 10 > "$L/03_classify_pilot.log" 2>&1
"$PY" - <<'EOF'
import json, sys
m = json.load(open("outputs/origin_placement_sensitivity/pilot/classify_manifest.json"))
bad = {k: m[k] for k in ["v1b_total_mismatch", "v1c_total_node_mismatch", "v1c_bridge_mismatch",
                         "inundation_cross_check_disagreements", "v3_total_issues"] if m[k] != 0}
bad.update({"v1a_mismatches": m["v1a"]["mismatches"]} if m["v1a"]["mismatches"] else {})
if bad:
    sys.exit(f"Pilot validation failed: {bad}")
print("pilot validation passed")
EOF
"$PY" "$D/04_summarize.py" --run pilot > "$L/04_summarize_pilot.log" 2>&1
"$PY" "$D/03_classify_draws.py" --subset all --n-draws 50 > "$L/03_classify_full.log" 2>&1
"$PY" "$D/04_summarize.py" --run full > "$L/04_summarize_full.log" 2>&1
"$RSCRIPT" "$D/05_bg_models.R" full > "$L/05_bg_models_full.log" 2>&1
"$PY" "$D/06_mechanism_diagnostics.py" > "$L/06_mechanism.log" 2>&1
"$PY" "$D/07_compare_models.py" --run full > "$L/07_compare_models.log" 2>&1
"$PY" "$D/08_write_run_manifest.py" > "$L/08_manifest.log" 2>&1
