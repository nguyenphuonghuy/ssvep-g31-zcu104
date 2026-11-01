#!/bin/bash
# bash run_power_std.sh <thu_muc_dau_ra> [REPS] [DUR] [DT]  (chay tren board, nen nohup)
OUT=$1; shift
mkdir -p "$OUT" && cd "$OUT" || exit 2
echo "START $(date '+%F %T') $* loadavg: $(cat /proc/loadavg)" > power_std.log
echo xilinx | sudo -S -E bash -c "source /etc/profile.d/pynq_venv.sh 2>/dev/null; source /etc/profile.d/xrt_setup.sh 2>/dev/null; \
  cd $PWD && taskset -c 0 python3 /home/xilinx/ssvep_lite_pkg_g31/power_std/power_std.py $*" >> power_std.log 2>&1
echo "rc=$? END $(date '+%F %T')" >> power_std.log
echo XONG > power_std.done
