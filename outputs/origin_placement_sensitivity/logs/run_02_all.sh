#!/usr/bin/env bash
cd /c/Users/Vivek/Dropbox/repos/slr_fl_fragile_access
export MKL_THREADING_LAYER=SEQUENTIAL GDAL_DATA=/c/Users/Vivek/miniforge3/envs/research-geo/Library/share/gdal PYTHONIOENCODING=utf-8
for s in 0 1 2 3 4 5 6; do
  /c/Users/Vivek/miniforge3/envs/research-geo/python.exe scripts/origin_placement_sensitivity/02_build_node_states.py --scenario $s > outputs/origin_placement_sensitivity/logs/02_node_states_${s}ft.log 2>&1
  echo "scenario $s EXIT $?"
done
