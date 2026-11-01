#!/bin/bash
#  buoc 0: do tim cam bien (chi doc). Chay tren board: bash probe.sh > probe.log 2>&1
echo "== uname: $(uname -a)"; echo "== loadavg: $(cat /proc/loadavg)"
echo "== cpufreq"
for c in /sys/devices/system/cpu/cpu[0-3]/cpufreq; do
  [ -d $c ] && echo "$c: gov=$(cat $c/scaling_governor) cur=$(cat $c/scaling_cur_freq) avail_gov=$(cat $c/scaling_available_governors 2>/dev/null) avail_f=$(cat $c/scaling_available_frequencies 2>/dev/null)"
done
echo "== hwmon"
for h in /sys/class/hwmon/hwmon*; do
  echo "--- $h name=$(cat $h/name 2>/dev/null) dev=$(readlink -f $h/device 2>/dev/null)"
  for f in $h/*_label; do [ -f "$f" ] && echo "  $(basename $f)=$(cat $f)"; done
  for f in $h/{in,curr,power,temp}*_input; do [ -f "$f" ] && echo "  $(basename $f)=$(cat $f)"; done
  [ -f $h/update_interval ] && echo "  update_interval=$(cat $h/update_interval)"
done
echo "== iio (sysmon)"
for d in /sys/bus/iio/devices/iio:device*; do
  echo "--- $d name=$(cat $d/name 2>/dev/null)"; ls $d | grep -E "temp|in_" | head -40 | tr '\n' ' '; echo
done
echo "== lap mau power*_input 40 lan, 50 ms (do phan giai, chu ky cap nhat)"
P=$(ls /sys/class/hwmon/hwmon*/power*_input 2>/dev/null)
for i in $(seq 1 40); do echo "$(date +%s.%N) $(for f in $P; do cat $f; done | tr '\n' ' ')"; sleep 0.05; done
echo "== cot: $P"
