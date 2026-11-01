#!/bin/bash
# Chay TREN BOARD: bash run_ens_board.sh [R=200] [N_RUNS=7] [CD=10]
# 1. bien dich A (-O2) va B (-O3 -mcpu=cortex-a53); 2. cong 1-3 cho A, B, NumPy (dung neu A khong dat);
# 3. do tre: moi ban (A, B, NumPy) x moi cua so: N_RUNS luot x R quyet dinh, nghi CD s.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 2
R=${1:-200}; N=${2:-7}; CD=${3:-10}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source /etc/profile.d/pynq_venv.sh 2>/dev/null
echo "===== START $(date '+%F %T') $(hostname) $(uname -m) g++ $(g++ -dumpfullversion) R=$R N=$N CD=$CD ====="
echo "loadavg: $(cat /proc/loadavg)"
python3 -c "import numpy; numpy.show_config()" 2>&1 | head -30
ldd "$(python3 -c 'import numpy.core._multiarray_umath as m; print(m.__file__)')" | grep -iE "blas|lapack"
g++ -std=c++17 -O2 ensdnn.cpp -o ensdnn_A || { echo "LOI bien dich A"; exit 2; }
g++ -std=c++17 -O3 -mcpu=cortex-a53 ensdnn.cpp -o ensdnn_B || { echo "LOI bien dich B"; exit 2; }
for T in 100 250; do
  ./ensdnn_A T$T check | tail -5; [ ${PIPESTATUS[0]} = 0 ] || { echo "A T$T: CONG KHONG DAT -> dung"; exit 3; }
  ./ensdnn_B T$T check | tail -5 | sed 's/^/[B] /'
  python3 ensdnn_np.py T$T check | sed 's/^/[NP] /'
done
echo "CONG A: DAT"
sleep "$CD"
for T in 100 250; do
  for v in A B NP; do
    for i in $(seq 1 "$N"); do
      echo "loadavg: $(cat /proc/loadavg)"
      if [ $v = NP ]; then python3 ensdnn_np.py T$T bench "$R" | grep BENCH | sed "s/^/run=$i v=$v /"
      else ./ensdnn_$v T$T bench "$R" | grep BENCH | sed "s/^/run=$i v=$v /"; cp T$T/bench_us.bin T$T/bench_us_${v}_run$i.bin; fi
      sleep "$CD"
    done
  done
done
echo "ENS_DONE $(date '+%F %T')"
