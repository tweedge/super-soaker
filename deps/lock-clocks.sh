#!/usr/bin/env bash
# deps/lock-clocks.sh — lock the Jetson into its highest power mode and max clocks.
# Mode id for MAXN_SUPER varies per nvpmodel.conf, so resolve it by name.
set -euo pipefail

MODE_ID="$(awk -F'[= >]' '/POWER_MODEL/ && /NAME=MAXN_SUPER/ {for(i=1;i<=NF;i++) if($i=="ID") print $(i+1)}' /etc/nvpmodel.conf | head -1)"
# This unit starts early (DefaultDependencies=no): both nvpmodel (its MAXN
# mode writes /sys/devices/platform/gpu.0/*_pg_mask) and jetson_clocks need
# the GPU sysfs to exist. Wait up to 60 s before touching either.
for _ in $(seq 1 60); do
    if [ -e /sys/devices/platform/gpu.0/tpc_pg_mask ]; then
        break
    fi
    sleep 1
done

if [ -z "${MODE_ID}" ]; then
    # fall back to whatever mode is the highest-power one currently set
    MODE_ID="$(nvpmodel -q | awk '/NV Power Mode/{getline; print; exit}')"
    echo "MAXN_SUPER not found in nvpmodel.conf; keeping mode ${MODE_ID}"
else
    echo "nvpmodel -m ${MODE_ID} (MAXN_SUPER)"
    nvpmodel -m "${MODE_ID}"
fi

jetson_clocks
nvpmodel -q