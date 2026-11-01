#!/bin/bash
cd ~/ssvep_lite_hls || exit 2
source /tools/Xilinx/Vivado/2024.2/settings64.sh
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 nice -n 19 vivado -mode batch -nojournal -source reports.tcl -tclargs $1 > logs/reports_$1.log 2>&1
echo "rc=$?" >> logs/reports_$1.log
