#!/bin/bash
# overlay v2 rieng T100 roi T250, tran RAM 8 GB, nice 19; OOM/loi thi dung, khong thu lai.
cd ~/ssvep_lite_hls || exit 2
bash run_vivado.sh 200 100 rt
grep -q "OVERLAY_DONE" logs/vivado_200_rtT100.log || { echo "T100 KHONG XONG -> dung" > logs/vivado_rt_seq.done; exit 1; }
bash run_vivado.sh 200 250 rt
grep -q "OVERLAY_DONE" logs/vivado_200_rtT250.log && echo "T100 + T250 XONG" > logs/vivado_rt_seq.done || echo "T250 KHONG XONG" > logs/vivado_rt_seq.done
