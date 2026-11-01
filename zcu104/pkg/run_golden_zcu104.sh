#!/bin/bash
# Chay TREN BOARD ZCU104: bash run_golden_zcu104.sh [N_RUNS=7] [COOLDOWN=10] [R=2000]
# 1. bien dich native + so khop tung bit voi expected_*; 2. do tre giai ma 1 trial (chi khi buoc 1 dat).
set -u
cd ~/ssvep_lite_pkg || exit 2
N=${1:-7}; CD=${2:-10}; R=${3:-2000}
echo "===== START $(date '+%F %T') $(hostname) $(uname -m) g++ $(g++ -dumpfullversion) N=$N CD=$CD R=$R ====="
echo "loadavg: $(cat /proc/loadavg)"
for T in 100 250; do
  g++ -std=c++20 -O2 -include T$T/params.h golden.cpp -o golden_T$T || { echo "LOI bien dich T$T"; exit 2; }
  ./golden_T$T T$T 8 || { echo "LOI chay T$T"; exit 2; }
  ok=1
  for k in x8 z N phi score pred; do
    if cmp -s T$T/out_$k.bin T$T/expected_$k.bin; then echo "T$T $k OK"; else echo "T$T $k KHAC"; ok=0; fi
  done
  if [ $ok = 1 ]; then echo "T$T KHOP TUNG BIT: DAT"; else echo "T$T KHOP TUNG BIT: KHONG DAT"; exit 3; fi
done
sleep "$CD"
for T in 100 250; do
  for i in $(seq 1 "$N"); do
    echo "loadavg: $(cat /proc/loadavg)"
    ./golden_T$T T$T 8 "$R" | grep BENCH | sed "s/^/run=$i /"
    rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { echo "LOI do T$T run $i rc=$rc"; exit 4; }
    cp T$T/bench_us.bin T$T/bench_us_run$i.bin
    sleep "$CD"
  done
done
echo "GOLDEN_DONE $(date '+%F %T')"
