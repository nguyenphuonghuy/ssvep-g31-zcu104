// Kernel HLS G3 v2.2 / G3.1 v2.3: tham so NAP KHI CHAY. v2.3: neu dims.h co #define G3_DC
// thi tru TB doan truoc kich thich [0, DC_ON) cho tung kenh truoc bank IIR (x' = sat16(x - dc)); khong co -> giong v2.2. Phep tinh GIONG HET g3_decode.cpp (v1, muc 13);
// chi khac: tham so nam trong mang static (BRAM/thanh ghi), nap tu DDR o che do mode = 0; mode = 1 giai ma 1 luot thu.
// Kich thuoc co dinh theo cua so: #include "dims.h" (export_hw.py). Chi dung kieu C chuan; moi phep dich >= 0.
#include <cstdint>
#include "dims.h"
#include "g3_decode_rt.h"

static constexpr int IN_SHIFT = 6, D_SHIFT = 10, D_BITS = 27, U_FRAC = 8;
static constexpr int NBC = NB * C, TT = S0 + T, NF = NB * F;
// vi tri cac mang trong khoi PRM (bo cuc export_hw.py)
static constexpr int OFF_NSEC = 9, OFF_SB = OFF_NSEC + NB, OFF_SA = OFF_SB + NB * MAXSEC * 3, OFF_SF = OFF_SA + NB * MAXSEC * 2,
                     OFF_W8 = OFF_SF + NB * MAXSEC, OFF_A8 = OFF_W8 + F * C, OFF_P8 = OFF_A8 + D,
                     OFF_R8 = OFF_P8 + NB * K * F * T, OFF_END = OFF_R8 + K * J * T;
static_assert(OFF_END == PRM_LEN, "bo cuc PRM khong khop dims.h");

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
int64_t g3_dbg_phi[K][D];   // chi C-sim
#endif

void g3_decode_rt(const int16_t in[C * TT], const int32_t prm[PRM_LEN], int64_t score[K], int32_t pred[1],
                  int32_t status[1], int32_t mode) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem depth = C * TT
#pragma HLS INTERFACE m_axi port = prm offset = slave bundle = gmem depth = PRM_LEN
#pragma HLS INTERFACE m_axi port = score offset = slave bundle = gmem depth = K
#pragma HLS INTERFACE m_axi port = pred offset = slave bundle = gmem depth = 1
#pragma HLS INTERFACE m_axi port = status offset = slave bundle = gmem depth = 1
#pragma HLS INTERFACE s_axilite port = mode
#pragma HLS INTERFACE s_axilite port = return

  // ---- tham so (giu giua cac lan goi) ----
  static int32_t MX, SHX, KTM, SHTM, KE, SHE, BINT;
  static int32_t NSEC[NB], SOS_B[NB][MAXSEC][3], SOS_A[NB][MAXSEC][2], SOS_FB[NB][MAXSEC];
  static int8_t W8[F][C], A8[D], P8[NB][K][F][T], R8[K][J][T];
#pragma HLS ARRAY_PARTITION variable = SOS_B dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = SOS_B dim = 3 complete
#pragma HLS ARRAY_PARTITION variable = SOS_A dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = SOS_A dim = 3 complete
#pragma HLS ARRAY_PARTITION variable = SOS_FB dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = W8 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = P8 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = P8 dim = 3 complete
#pragma HLS ARRAY_PARTITION variable = R8 dim = 2 complete
#pragma HLS BIND_STORAGE variable = P8 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = R8 type = ram_2p impl = bram

  if (mode == 0) {
    // ---- nap tham so (v2.2): MOT vong pipeline doc tuan tu prm[0..PRM_LEN) (1 luong burst), ghi vao mang bang bo dem
    //      tang dan. Ban v2.1 (doc tung phan tu o vong khong pipeline) cho gia tri -1 tren phan cung thuc. ----
    int32_t magic = 0, tt = 0;
    int a = 0, b = 0, c = 0, d = 0;                  // bo dem chi so trong doan hien tai (reset khi sang doan moi)
    for (int i = 0; i < PRM_LEN; ++i) {
#pragma HLS PIPELINE II = 1
      const int32_t v = prm[i];
      if (i == OFF_NSEC || i == OFF_SB || i == OFF_SA || i == OFF_SF || i == OFF_W8 || i == OFF_A8 || i == OFF_P8 ||
          i == OFF_R8) { a = 0; b = 0; c = 0; d = 0; }
      if (i < OFF_NSEC) {
        if (i == 0) magic = v;
        else if (i == 1) tt = v;
        else if (i == 2) MX = v;
        else if (i == 3) SHX = v;
        else if (i == 4) KTM = v;
        else if (i == 5) SHTM = v;
        else if (i == 6) KE = v;
        else if (i == 7) SHE = v;
        else BINT = v;
      } else if (i < OFF_SB) {
        NSEC[a] = v; ++a;                                            // a = n
      } else if (i < OFF_SA) {
        SOS_B[a][b][c] = v;                                          // (n, s, k3)
        if (c == 2) { c = 0; if (b == MAXSEC - 1) { b = 0; ++a; } else ++b; } else ++c;
      } else if (i < OFF_SF) {
        SOS_A[a][b][c] = v;                                          // (n, s, k2)
        if (c == 1) { c = 0; if (b == MAXSEC - 1) { b = 0; ++a; } else ++b; } else ++c;
      } else if (i < OFF_W8) {
        SOS_FB[a][b] = v;                                            // (n, s)
        if (b == MAXSEC - 1) { b = 0; ++a; } else ++b;
      } else if (i < OFF_A8) {
        W8[a][b] = int8_t(v);                                        // (f, c)
        if (b == C - 1) { b = 0; ++a; } else ++b;
      } else if (i < OFF_P8) {
        A8[a] = int8_t(v); ++a;                                      // d
      } else if (i < OFF_R8) {
        P8[a][b][c][d] = int8_t(v);                                  // (n, k, f, t)
        if (d == T - 1) { d = 0; if (c == F - 1) { c = 0; if (b == K - 1) { b = 0; ++a; } else ++b; } else ++c; } else ++d;
      } else {
        R8[a][b][c] = int8_t(v);                                     // (k, j, t)
        if (c == T - 1) { c = 0; if (b == J - 1) { b = 0; ++a; } else ++b; } else ++c;
      }
    }
    // ---- checksum DOC NGUOC tu mang da luu: cs = sum v * ((chi so trong khoi % 1021) + 1), 5 doan -> score[0..4] ----
    int64_t cs0 = 0, cs1 = 0, cs2 = 0, cs3 = 0, cs4 = 0;
    {
      const int32_t hdr[9] = {magic, tt, MX, SHX, KTM, SHTM, KE, SHE, BINT};
      for (int i = 0; i < 9; ++i) cs0 += int64_t(hdr[i]) * ((i % 1021) + 1);
      for (int n = 0; n < NB; ++n) cs0 += int64_t(NSEC[n]) * (((OFF_NSEC + n) % 1021) + 1);
      for (int n = 0; n < NB; ++n)
        for (int s = 0; s < MAXSEC; ++s) {
          for (int i = 0; i < 3; ++i) cs0 += int64_t(SOS_B[n][s][i]) * (((OFF_SB + (n * MAXSEC + s) * 3 + i) % 1021) + 1);
          for (int i = 0; i < 2; ++i) cs0 += int64_t(SOS_A[n][s][i]) * (((OFF_SA + (n * MAXSEC + s) * 2 + i) % 1021) + 1);
          cs0 += int64_t(SOS_FB[n][s]) * (((OFF_SF + n * MAXSEC + s) % 1021) + 1);
        }
      for (int f = 0; f < F; ++f)
        for (int c = 0; c < C; ++c) cs1 += int64_t(W8[f][c]) * (((OFF_W8 + f * C + c) % 1021) + 1);
      for (int d = 0; d < D; ++d) cs2 += int64_t(A8[d]) * (((OFF_A8 + d) % 1021) + 1);
      for (int n = 0; n < NB; ++n)
        for (int k = 0; k < K; ++k)
          for (int f = 0; f < F; ++f)
            for (int t = 0; t < T; ++t) {
#pragma HLS PIPELINE II = 1
              cs3 += int64_t(P8[n][k][f][t]) * (((OFF_P8 + ((n * K + k) * F + f) * T + t) % 1021) + 1);
            }
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < J; ++j)
          for (int t = 0; t < T; ++t) {
#pragma HLS PIPELINE II = 1
            cs4 += int64_t(R8[k][j][t]) * (((OFF_R8 + (k * J + j) * T + t) % 1021) + 1);
          }
    }
    score[0] = cs0; score[1] = cs1; score[2] = cs2; score[3] = cs3; score[4] = cs4;
    status[0] = (magic == PRM_MAGIC && tt == T) ? 1 : -1;
    return;
  }

  // ---- 0. doc dau vao ----
  int16_t xin[C][TT];
#pragma HLS ARRAY_PARTITION variable = xin dim = 1 complete
#ifdef G3_DC
  int32_t dcv[C];
#pragma HLS ARRAY_PARTITION variable = dcv complete
#endif
  for (int c = 0; c < C; ++c) {
#ifdef G3_DC
    int32_t S = 0;                                   // |S| <= DC_ON * 2^15 (< 2^23 voi DC_ON = 125)
#endif
    for (int t = 0; t < TT; ++t) {
#pragma HLS PIPELINE II = 1
      const int16_t v = in[c * TT + t];
      xin[c][t] = v;
#ifdef G3_DC
      if (t < DC_ON) S += v;
#endif
    }
#ifdef G3_DC
    dcv[c] = int32_t(rsr(int64_t(S) * MDC, int(SHDC)));
#endif
  }

  // ---- 1. bank IIR + 2. luong tu INT8 ----
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
#ifdef G3_DC
      int32_t v = int32_t(sat(int64_t(xin[c][t]) - dcv[c], 16)) << (D_SHIFT - IN_SHIFT);
#else
      int32_t v = int32_t(xin[c][t]) << (D_SHIFT - IN_SHIFT);
#endif
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
      const int q = (b - 29) >> 1;
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
  status[0] = 2;
}
