#!/bin/bash
#  : do tre (run_ens_board.sh) roi cong suat (ens_power.py), noi tiep.
cd "$(dirname "$(readlink -f "$0")")" || exit 2
bash run_ens_board.sh 200 7 10 > ens_board.log 2>&1
grep -q ENS_DONE ens_board.log || { echo "ENS_BOARD_LOI" > run_ens_all.done; exit 3; }
bash run_pl.sh ens_power.py ens_power.log 3 60 0.1
grep -q ENS_POWER_DONE ens_power.log && echo XONG > run_ens_all.done || echo POWER_LOI > run_ens_all.done
