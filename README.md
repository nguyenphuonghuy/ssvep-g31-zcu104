# ssvep-g31-zcu104

Code for the paper *Hardware-Efficient Calibration-Free SSVEP Decoding on Zynq UltraScale+: A Hybrid Correlation–Deep Learning Approach*
(D. T. M. Thuong and N. P. Huy, submitted to IEEE Transactions on Biomedical Circuits and Systems).

The repository contains the training and evaluation code of the hybrid decoder (G3.1), the fixed-point reference model,
the Vitis HLS kernel for the ZCU104 programmable logic, and the board-side latency and energy measurement scripts.

## Layout

| Path | Content |
|---|---|
| `scripts/` | Data loading, causal filter bank, quantization-aware training (`train_loso.py`), baselines (`baselines_tf.py`, `wp3_fbcca.py`), Ensemble-DNN reproduction (`ensemble_dnn.py`), statistical analyses (`analyze_*.py`) |
| `scripts/golden/` | Bit-exact fixed-point reference model (Python) and parameter export for the hardware |
| `scripts/zcu104/` | C++ ports used for cost measurements on the Arm Cortex-A53 |
| `hls/` | HLS kernel (`g3_decode_rt.cpp`), testbenches and Vivado build scripts (`stage_g31/` for G3.1) |
| `zcu104/` | PYNQ drivers, auto-restart power measurement (`power_std/`), latency and energy scripts |
| `configs/datasets.json` | Dataset definitions (channels, latency, pre-stimulus length) |

## Data

All four datasets are public: Tsinghua Benchmark, BETA, Nakanishi et al. (2015) and the wearable dataset of Zhu et al. (2021).
Download them and set `SSVEP_DATA` to the folder that contains `benchmark/`, `beta/`, `nakanishi2015/` and `wearable_zhu2021/`.
Set `SSVEP_ROOT` to the repository root if the scripts are run from elsewhere.

## Requirements

Python 3.11 with the packages in `requirements.txt` (PyTorch 2.7). Hardware: Vitis HLS and Vivado 2024.2, a ZCU104 board with PYNQ 3.0.1.

## Typical use

```bash
export SSVEP_DATA=/path/to/datasets
python scripts/train_loso.py --dataset benchmark --model wp3_g3_dc --seeds 0 --windows 0.4 --n_val 4 --out_dir results/g31
python scripts/golden/golden_eval.py --help      # fixed-point reference model
```

## License

MIT, see `LICENSE`.
