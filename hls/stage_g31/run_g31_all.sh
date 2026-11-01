#!/bin/bash
# G3.1 v2.3: HLS T100+T250 roi Vivado 2 overlay rieng. Chi chay khi may ranh; tran 10 GB; loi thi dung.
cd ~/ssvep_lite_hls_g31 || exit 2
L=logs/g31_all.log; echo "CHO $(date '+%F %T')" > $L
cho_ranh() {   # chay TRUOC MOI buoc: khong co vitis_hls/vivado nao va >= 10 GB trong (= tran MemoryMax)
  while true; do
    busy=$(ps -eo comm | grep -ciE "^(vitis_hls|vivado|loader|vitis)$"); av=$(free -m | awk '/Mem/{print $7}')
    [ "$busy" -eq 0 ] && [ "$av" -ge 10000 ] && { echo "$(date +%T) ranh -> $1" >> $L; return; }
    echo "$(date +%T) ban: $busy tien trinh, avail ${av}MB" >> $L; sleep 300
  done
}
for T in 100 250; do
  cho_ranh "HLS T$T"
  nice -n 19 bash run_hls_rt.sh $T "csim csynth cosim export"
  f=logs/hls_rt_T${T}_csim_csynth_cosim_export.log
  grep -q "co-simulation finished: PASS" $f || { echo "HLS T$T KHONG DAT -> dung" >> $L; echo "XONG_LOI" >> $L; exit 1; }
  cho_ranh "Vivado T$T"
  bash run_vivado.sh 200 $T rt
  grep -q OVERLAY_DONE logs/vivado_200_rtT$T.log || { echo "Vivado T$T KHONG XONG -> dung" >> $L; echo "XONG_LOI" >> $L; exit 1; }
  echo "T$T XONG $(date +%T)" >> $L
done
echo "XONG_DAT $(date '+%F %T')" >> $L
