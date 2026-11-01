#!/bin/bash
# Chay tren board: bash run_pl.sh <script.py> <log> [tham so...]  (vd: run_pl.sh g3_pl.py pl.log 7 2000 10 g3_T100.bit)
cd "$(dirname "$(readlink -f "$0")")" || exit 2
HERE=$PWD
PYS=$1; LOG=$2; shift 2
echo "START $(date '+%F %T') $PYS $* loadavg: $(cat /proc/loadavg)" > $LOG
echo xilinx | sudo -S -E bash -c "source /etc/profile.d/pynq_venv.sh 2>/dev/null; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \
  cd $HERE && python3 $PYS $*" >> $LOG 2>&1
echo "rc=$? END $(date '+%F %T')" >> $LOG
