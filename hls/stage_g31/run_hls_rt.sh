#!/bin/bash
# Chay tren may host: bash run_hls.sh <T> "<buoc>"   (log: logs/hls_T<T>_<buoc>.log)
set -u
cd ~/ssvep_lite_hls_g31 || exit 2
T=$1; STEPS=$2
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh 2>/dev/null || source /tools/Xilinx/Vitis/2024.2/settings64.sh
mkdir -p logs
TAG=$(echo "$STEPS" | tr ' ' '_')
echo "START $(date '+%F %T') T=$T STEPS=$STEPS" > logs/hls_rt_T${T}_$TAG.log
G3_T=$T G3_STEPS="$STEPS" vitis_hls -f run_hls_rt.tcl >> logs/hls_rt_T${T}_$TAG.log 2>&1
echo "rc=$? END $(date '+%F %T')" >> logs/hls_rt_T${T}_$TAG.log
