"""Chi so danh gia. Chi dung thu vien chuan de chay duoc tren login node."""
import math


def itr_bpm(p, n_targets, win_s, gaze_s):
    """ITR (bits/phut). T = cua so + thoi gian chuyen anh nhin. p <= 1/M -> 0."""
    m = n_targets
    if p <= 1.0 / m:
        return 0.0
    bits = math.log2(m)
    if p < 1.0:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (m - 1))
    return bits * 60.0 / (win_s + gaze_s)
