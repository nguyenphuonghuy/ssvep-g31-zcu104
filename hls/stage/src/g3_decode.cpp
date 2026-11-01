// Kernel HLS cua bo giai ma G3 fixed-point. Phai khop tung bit voi
// scripts/golden/golden.cpp va golden_model.py. Chi dung kieu C chuan (int8/16/32/64) de bien dich duoc
// bang g++ (kiem tren HPC) va Vitis HLS. Moi phep dich deu co so dich >= 0 (khong can C++20).
// Tham so: #include "params.h" (sinh boi golden_eval.py, theo T va fold).
#include <cstdint>
#include "params.h"
#include "g3_decode.h"

static constexpr int IN_SHIFT = 6, D_SHIFT = 10, D_BITS = 27, U_FRAC = 8;
static constexpr int NBC = NB * C, TT = S0 + T, NF = NB * F;

static inline int64_t rsr(int64_t v, int s) {
#pragma HLS INLINE
  if (s > 0) return (v + (int64_t(1) << (s - 1))) >> s;
  return v << (-s);
}
static inline int64_t sat(int64_t v, int bits) {
#pragma HLS INLINE
  const int64_t lo = -(int64_t(1) << (bits - 1)), hi = (int64_t(1) << (bits - 1)) - 1;
  return v < lo ? lo : (v > hi ? hi : v);
}

#ifndef __SYNTHESIS__
int64_t g3_dbg_phi[K][D];   // chi C-sim: doi chieu phi voi expected_phi
#endif

void g3_decode(const int16_t in[C * TT], int64_t score[K], int32_t pred[1]) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = C * TT
#pragma HLS INTERFACE m_axi port = score offset = slave bundle = gmem depth = K
#pragma HLS INTERFACE m_axi port = pred offset = slave bundle = gmem depth = 1
#pragma HLS INTERFACE s_axilite port = return

  // ---- 0. doc dau vao ----
  int16_t xin[C][TT];
#pragma HLS ARRAY_PARTITION variable = xin dim = 1 complete
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < TT; ++t) {
#pragma HLS PIPELINE II = 1
      xin[c][t] = in[c * TT + t];
    }

  // ---- 1. bank IIR (DF-I, section noi tiep) + 2. luong tu INT8 ----
  int32_t x1[MAXSEC][NBC], x2[MAXSEC][NBC], y1[MAXSEC][NBC], y2[MAXSEC][NBC];
#pragma HLS ARRAY_PARTITION variable = x1 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = x2 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = y1 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = y2 dim = 1 complete
  for (int s = 0; s < MAXSEC; ++s)
    for (int i = 0; i < NBC; ++i) {
#pragma HLS PIPELINE II = 1
      x1[s][i] = 0; x2[s][i] = 0; y1[s][i] = 0; y2[s][i] = 0;
    }
  int8_t x8[NB][C][T];
#pragma HLS ARRAY_PARTITION variable = x8 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = x8 dim = 2 complete
  for (int t = 0; t < TT; ++t) {
    for (int i = 0; i < NBC; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = x1 type = inter false
#pragma HLS DEPENDENCE variable = x2 type = inter false
#pragma HLS DEPENDENCE variable = y1 type = inter false
#pragma HLS DEPENDENCE variable = y2 type = inter false
      const int n = i / C, c = i % C;
      int32_t v = int32_t(xin[c][t]) << (D_SHIFT - IN_SHIFT);
      for (int s = 0; s < MAXSEC; ++s) {
#pragma HLS UNROLL
        if (s < NSEC[n]) {
          const int64_t accb = SOS_B[n][s][0] * int64_t(v) + SOS_B[n][s][1] * int64_t(x1[s][i]) +
                               SOS_B[n][s][2] * int64_t(x2[s][i]);
          const int64_t acc = rsr(accb, int(SOS_FB[n][s]) - A_FRAC) - SOS_A[n][s][0] * int64_t(y1[s][i]) -
                              SOS_A[n][s][1] * int64_t(y2[s][i]);
          const int32_t yt = int32_t(sat(rsr(acc, A_FRAC), D_BITS));
          x2[s][i] = x1[s][i]; x1[s][i] = v; y2[s][i] = y1[s][i]; y1[s][i] = yt;
          v = yt;
        }
      }
      if (t >= S0) x8[n][c][t - S0] = int8_t(sat(rsr(int64_t(v) * MX, SHX), 8));
    }
  }

  // ---- 3. loc khong gian + chuan N ----
  int32_t z[NB][F][T];
#pragma HLS ARRAY_PARTITION variable = z dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = z dim = 2 complete
  int64_t N[NB][F];
#pragma HLS ARRAY_PARTITION variable = N complete dim = 0
  for (int n = 0; n < NB; ++n) {
    int64_t nn[F];
#pragma HLS ARRAY_PARTITION variable = nn complete
    for (int f = 0; f < F; ++f) nn[f] = 0;
    for (int t = 0; t < T; ++t) {
#pragma HLS PIPELINE II = 1
      for (int f = 0; f < F; ++f) {
        int32_t acc = 0;
        for (int c = 0; c < C; ++c) acc += int32_t(W8[f][c]) * int32_t(x8[n][c][t]);
        z[n][f][t] = acc;
        nn[f] += int64_t(acc) * acc;
      }
    }
    for (int f = 0; f < F; ++f) N[n][f] = nn[f];
  }

  // ---- 4. 1/sqrt(N) bang LUT ----
  int32_t lut[NB][F], shv[NB][F];
#pragma HLS ARRAY_PARTITION variable = lut complete dim = 0
#pragma HLS ARRAY_PARTITION variable = shv complete dim = 0
  for (int n = 0; n < NB; ++n)
    for (int f = 0; f < F; ++f) {
#pragma HLS PIPELINE II = 1
      const int64_t nv = N[n][f];
      const int64_t nz = nv > 0 ? nv : 1;
      int b = 1;
      for (int k = 0; k < 63; ++k)
        if ((nz >> k) & 1) b = k + 1;
      const int q = (b - 29) >> 1;   // floor((b-29)/2): dich so hoc
      const int64_t m = q >= 0 ? (nz >> (2 * q)) : (nz << (-2 * q));
      const int idx = int(m >> 20);
      lut[n][f] = nv > 0 ? RSQRT_LUT[idx - 256] : 0;
      shv[n][f] = 30 + q - U_FRAC;
    }

  // ---- 5. tuong quan, dac trung, ham diem ----
  int64_t best_s = 0;
  int32_t best_k = 0;
  for (int k = 0; k < K; ++k) {
    int64_t ctm[NB][F], csc[NB][F][J];
#pragma HLS ARRAY_PARTITION variable = ctm complete dim = 0
#pragma HLS ARRAY_PARTITION variable = csc complete dim = 0
    for (int n = 0; n < NB; ++n)
      for (int f = 0; f < F; ++f) {
#pragma HLS UNROLL
        ctm[n][f] = 0;
        for (int j = 0; j < J; ++j) csc[n][f][j] = 0;
      }
    for (int t = 0; t < T; ++t) {
#pragma HLS PIPELINE II = 1
      for (int n = 0; n < NB; ++n)
        for (int f = 0; f < F; ++f) {
          const int32_t zz = z[n][f][t];
          ctm[n][f] += int64_t(zz * int32_t(P8[n][k][f][t]));
          for (int j = 0; j < J; ++j) csc[n][f][j] += int64_t(zz * int32_t(R8[k][j][t]));
        }
    }
    int8_t phi[D];
#pragma HLS ARRAY_PARTITION variable = phi complete
    for (int n = 0; n < NB; ++n)
      for (int f = 0; f < F; ++f) {
#pragma HLS PIPELINE II = 1
        const int64_t L = lut[n][f];
        const int sh = shv[n][f];
        const int64_t utm = rsr(ctm[n][f] * L, sh);
        phi[n * F + f] = int8_t(sat(rsr(utm * KTM, SHTM), 8));
        for (int h = 0; h < NH; ++h) {
          const int64_t us = rsr(csc[n][f][2 * h] * L, sh), uc = rsr(csc[n][f][2 * h + 1] * L, sh);
          phi[NF + (n * F + f) * NH + h] = int8_t(sat(rsr((us * us + uc * uc) * KE, SHE), 8));
        }
      }
    int64_t s = BINT;
    for (int d = 0; d < D; ++d) {
#pragma HLS PIPELINE II = 1
      s += int64_t(A8[d]) * int64_t(phi[d]);
    }
#ifndef __SYNTHESIS__
    for (int d = 0; d < D; ++d) g3_dbg_phi[k][d] = phi[d];
#endif
    score[k] = s;
    if (k == 0 || s > best_s) { best_s = s; best_k = k; }
  }
  pred[0] = best_k;
}
