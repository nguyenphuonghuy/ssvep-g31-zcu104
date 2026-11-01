#!/bin/bash
# Chay tren may host: bash run_vivado.sh <PL_MHZ> "<T...>"   (log: logs/vivado_<MHZ>_T<T..>.log)
# Gioi han tai: nice 19 + tran RAM 8 GB (systemd-run --user, neu co) — lan dau -jobs 8 lam treo may (2026-10-14).
cd ~/ssvep_lite_hls_g31 || exit 2
source /tools/Xilinx/Vivado/2024.2/settings64.sh
mkdir -p logs
LOG=logs/vivado_$1_${3:-}T$(echo $2 | tr ' ' '_').log
echo "START $(date '+%F %T') $(free -g | awk '/Mem/{print "RAM tong",$2,"GB, trong",$7,"GB"}')" > $LOG
CMD=(nice -n 19 vivado -mode batch -nojournal -source build_overlay.tcl -tclargs "$1" "$2" ${3:-})
if systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 true >/dev/null 2>&1; then
  echo "gioi han: systemd-run MemoryMax=10G" >> $LOG
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 "${CMD[@]}" >> $LOG 2>&1
else
  echo "gioi han: chi nice 19 (khong co systemd-run --user)" >> $LOG
  "${CMD[@]}" >> $LOG 2>&1
fi
echo "rc=$? END $(date '+%F %T')" >> $LOG
