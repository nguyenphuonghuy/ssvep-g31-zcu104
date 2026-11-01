#!/bin/bash
# Dung goi G3.1 cho may tong hop: hls/stage_g31 -> ~/ssvep_lite_hls_g31 (thu muc RIENG, khong dung v2.2).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); PROJ=$(dirname "$HERE")
HWV=$PROJ/results/hw_vectors_g31_2026_10_18
S=$HERE/stage_g31
[ -e "$S" ] && { echo "LOI: $S da ton tai"; exit 2; }
mkdir -p "$S/src_rt" "$S/logs"
cp "$HERE/g3_decode_rt.cpp" "$HERE/g3_decode_rt.h" "$HERE/tb_g3_rt.cpp" "$S/src_rt/"
for T in 100 250; do
  d=$S/vectors_rt/T$T; mkdir -p "$d"
  grep -q "define G3_DC" "$HWV/T$T/dims.h" || { echo "LOI: dims.h T$T thieu G3_DC"; exit 3; }
  cp "$HWV/T$T/dims.h" "$HWV/T$T/manifest.json" "$HWV/T$T/expected_phi.bin" "$d/"
  for f in 1 2; do for k in params input score pred label; do cp "$HWV/T$T/fold${f}_$k.bin" "$d/"; done; done
done
for f in run_hls_rt.sh run_hls_rt.tcl build_overlay.tcl run_vivado.sh run_rt22_all.sh; do
  sed 's|~/ssvep_lite_hls|~/ssvep_lite_hls_g31|g' "$HERE/$f" > "$S/$f"
done
mv "$S/run_rt22_all.sh" "$S/run_g31_all.sh"
sed -i -e 's|^# v2.2 .*|# G3.1 v2.3: HLS T100+T250 roi Vivado 2 overlay rieng. Chi chay khi may ranh; tran 10 GB; loi thi dung.|' \
       -e 's|logs/rt22_all.log|logs/g31_all.log|' -e '/vivado_200_rtT${T}_v21.log/d' "$S/run_g31_all.sh"
grep -n "ssvep_lite_hls" "$S"/*.sh "$S"/*.tcl
(cd "$S" && md5sum src_rt/* vectors_rt/T*/dims.h > MD5SUMS)
echo "STAGE OK: $S"
