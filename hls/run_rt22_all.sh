#!/bin/bash
# v2.2 (nap tham so mot luong): HLS T100+T250 roi Vivado 2 overlay rieng. Chi chay khi may ranh; tran 10 GB; loi thi dung.
cd ~/ssvep_lite_hls || exit 2
L=logs/rt22_all.log; echo "CHO $(date '+%F %T')" > $L
while true; do
  busy=$(ps -eo comm | grep -ciE "^(vitis_hls|vivado|loader|vitis)$"); av=$(free -m | awk '/Mem/{print $7}')
  [ "$busy" -eq 0 ] && [ "$av" -ge 11000 ] && break
  echo "$(date +%T) ban: $busy tien trinh, avail ${av}MB" >> $L; sleep 300
done
for T in 100 250; do
  nice -n 19 bash run_hls_rt.sh $T "csim csynth cosim export"
  f=logs/hls_rt_T${T}_csim_csynth_cosim_export.log
  grep -q "co-simulation finished: PASS" $f || { echo "HLS T$T KHONG DAT -> dung" >> $L; echo "XONG_LOI" >> $L; exit 1; }
  mv logs/vivado_200_rtT$T.log logs/vivado_200_rtT${T}_v21.log 2>/dev/null
  bash run_vivado.sh 200 $T rt
  grep -q OVERLAY_DONE logs/vivado_200_rtT$T.log || { echo "Vivado T$T KHONG XONG -> dung" >> $L; echo "XONG_LOI" >> $L; exit 1; }
  echo "T$T XONG $(date +%T)" >> $L
done
echo "XONG_DAT $(date '+%F %T')" >> $L
