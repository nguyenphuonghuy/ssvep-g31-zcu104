#!/bin/bash
# tran 10 GB, CHI chay khi may ranh (khong vitis_hls/vivado khac, RAM trong >= 11 GB);
# overlay v2 rieng T100 roi T250; OOM/loi thi dung, khong thu lai.
cd ~/ssvep_lite_hls || exit 2
echo "CHO $(date '+%F %T')" > logs/vivado_rt_wait.log
while true; do
  busy=$(ps -eo comm | grep -ciE "^(vitis_hls|vivado|loader|vitis)$")
  av=$(free -m | awk '/Mem/{print $7}')
  if [ "$busy" -eq 0 ] && [ "$av" -ge 11000 ]; then break; fi
  echo "$(date +%T) ban: tien trinh=$busy avail=${av}MB" >> logs/vivado_rt_wait.log
  sleep 300
done
echo "BAT DAU $(date '+%F %T') avail=$(free -m | awk '/Mem/{print $7}')MB" >> logs/vivado_rt_wait.log
bash run_vivado_rt_seq.sh
