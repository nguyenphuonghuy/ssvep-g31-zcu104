#!/bin/bash
# Chay TREN BOARD: bash run_tt_board.sh [R=200] [N_RUNS=7] [CD=10]
# 1. bien dich A (-O2), B (-O3 -mcpu=cortex-a53); 2. cong 1-3 cho A (dung neu khong dat), B mo ta;
# 3. do tre: moi ban (A, B) x phuong phap (tt, fb) x cua so: N_RUNS luot x R quyet dinh, nghi CD s.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
R=${1:-200}; N=${2:-7}; CD=${3:-10}
echo "===== START $(date '+%F %T') $(hostname) $(uname -m) g++ $(g++ -dumpfullversion) R=$R N=$N CD=$CD ====="
echo "loadavg: $(cat /proc/loadavg)"
g++ -std=c++17 -O2 ttcca.cpp -o ttcca_A || { echo "LOI bien dich A"; exit 2; }
g++ -std=c++17 -O3 -mcpu=cortex-a53 ttcca.cpp -o ttcca_B || { echo "LOI bien dich B"; exit 2; }
for T in 100 250; do
  ./ttcca_A T$T check | tail -5; [ ${PIPESTATUS[0]} = 0 ] || { echo "A T$T: CONG KHONG DAT -> dung"; exit 3; }
  ./ttcca_B T$T check | tail -5 | sed 's/^/[B] /'
done
echo "CONG A: DAT"
sleep "$CD"
for T in 100 250; do
  for v in A B; do
    for m in tt fb; do
      for i in $(seq 1 "$N"); do
        echo "loadavg: $(cat /proc/loadavg)"
        ./ttcca_$v T$T bench $m "$R" | grep BENCH | sed "s/^/run=$i v=$v /"
        cp T$T/bench_us_$m.bin T$T/bench_us_${v}_${m}_run$i.bin
        sleep "$CD"
      done
    done
  done
done
echo "TT_DONE $(date '+%F %T')"
