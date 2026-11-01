// Golden model fixed-point G3 — ban C++ phai khop tung bit voi golden_model.py.
// Bien dich: g++ -std=c++20 -O2 -include <vec_dir>/T<T>/params.h golden.cpp -o golden_T<T>
// Chay:      ./golden_T<T> <vec_dir>/T<T> <B>            (doc input.bin, ghi out_{x8,z,N,phi,score,pred}.bin)
// Do tre:    ./golden_T<T> <vec_dir>/T<T> <B> <so_lan>   (them: giai ma tung trial (B=1) <so_lan> lan, in thoi gian;
//            kiem pred tung trial trung pred chay ca lo; KHONG doi phep tinh)
// C++20: dich trai so am va dich phai so hoc da duoc dinh nghia.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr int IN_SHIFT = 6, D_SHIFT = 10, D_BITS = 27, U_FRAC = 8;   // A_FRAC lay tu params.h

static inline int64_t rsr(int64_t v, int64_t s) {
  if (s > 0) return (v + (int64_t(1) << (s - 1))) >> s;
  return v << (-s);
}
static inline int64_t sat(int64_t v, int bits) {
  const int64_t lo = -(int64_t(1) << (bits - 1)), hi = (int64_t(1) << (bits - 1)) - 1;
  return v < lo ? lo : (v > hi ? hi : v);
}
static inline int64_t floor_div2(int64_t a) { return a >= 0 ? a / 2 : -((-a + 1) / 2); }

static void inv_sqrt(int64_t n, int64_t &lut, int64_t &q) {
  int64_t nz = n > 0 ? n : 1;
  int64_t b = 64 - __builtin_clzll((unsigned long long)nz);
  q = floor_div2(b - 29);
  int64_t m = q >= 0 ? (nz >> (2 * q)) : (nz << (-2 * q));
  int64_t idx = m >> 20;
  lut = n > 0 ? RSQRT_LUT[idx - 256] : 0;
}

template <typename Tp>
static void dump(const std::string &path, const std::vector<Tp> &v) {
  FILE *f = std::fopen(path.c_str(), "wb");
  std::fwrite(v.data(), sizeof(Tp), v.size(), f);
  std::fclose(f);
}

struct Out { std::vector<int64_t> x8, z, N, phi, score, pred; };

// Giai ma B trial lien tiep trong in (B x C x (S0+T) int16) -> o. Phep tinh giu nguyen ban truoc (.bak_before_bench).
static void decode(const int16_t *in, int B, Out &o) {
  const int TT = S0 + T;
  auto &x8 = o.x8; auto &z = o.z; auto &N = o.N; auto &phi = o.phi; auto &score = o.score; auto &pred = o.pred;
  // 1. bank IIR + 2. luong tu INT8 (chi giu T mau cuoi)
  x8.assign((size_t)NB * B * C * T, 0);
  std::vector<int64_t> sig(TT), out(TT);
  for (int n = 0; n < NB; ++n)
    for (int b = 0; b < B; ++b)
      for (int c = 0; c < C; ++c) {
        for (int t = 0; t < TT; ++t) sig[t] = int64_t(in[((size_t)b * C + c) * TT + t]) << (D_SHIFT - IN_SHIFT);
        for (int s = 0; s < NSEC[n]; ++s) {
          const int64_t b0 = SOS_B[n][s][0], b1 = SOS_B[n][s][1], b2 = SOS_B[n][s][2];
          const int64_t a1 = SOS_A[n][s][0], a2 = SOS_A[n][s][1], fb = SOS_FB[n][s];
          int64_t x1 = 0, x2 = 0, y1 = 0, y2 = 0;
          for (int t = 0; t < TT; ++t) {
            const int64_t xt = sig[t];
            const int64_t accb = b0 * xt + b1 * x1 + b2 * x2;
            const int64_t acc = rsr(accb, fb - A_FRAC) - a1 * y1 - a2 * y2;
            const int64_t yt = sat(rsr(acc, A_FRAC), D_BITS);
            out[t] = yt;
            x2 = x1; x1 = xt; y2 = y1; y1 = yt;
          }
          sig.swap(out);
        }
        for (int t = 0; t < T; ++t)
          x8[(((size_t)n * B + b) * C + c) * T + t] = sat(rsr(sig[S0 + t] * MX, SHX), 8);
      }

  // 3. loc khong gian, chuan, tuong quan, dac trung, ham diem
  z.assign((size_t)B * NB * F * T, 0); N.assign((size_t)B * NB * F, 0); phi.assign((size_t)B * K * D, 0);
  score.assign((size_t)B * K, 0); pred.assign(B, 0);
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < NB; ++n)
      for (int fl = 0; fl < F; ++fl) {
        int64_t nn = 0;
        for (int t = 0; t < T; ++t) {
          int64_t acc = 0;
          for (int c = 0; c < C; ++c) acc += int64_t(W8[fl][c]) * x8[(((size_t)n * B + b) * C + c) * T + t];
          z[(((size_t)b * NB + n) * F + fl) * T + t] = acc;
          nn += acc * acc;
        }
        N[((size_t)b * NB + n) * F + fl] = nn;
      }
    for (int k = 0; k < K; ++k) {
      int64_t *ph = &phi[((size_t)b * K + k) * D];
      for (int n = 0; n < NB; ++n)
        for (int fl = 0; fl < F; ++fl) {
          int64_t lut, q;
          inv_sqrt(N[((size_t)b * NB + n) * F + fl], lut, q);
          const int64_t sh = 30 + q - U_FRAC;
          const int64_t *zz = &z[(((size_t)b * NB + n) * F + fl) * T];
          int64_t ctm = 0;
          for (int t = 0; t < T; ++t) ctm += zz[t] * int64_t(P8[n][k][fl][t]);
          const int64_t utm = rsr(ctm * lut, sh);
          ph[n * F + fl] = sat(rsr(utm * KTM, SHTM), 8);
          for (int h = 0; h < NH; ++h) {
            int64_t cs = 0, cc = 0;
            for (int t = 0; t < T; ++t) {
              cs += zz[t] * int64_t(R8[k][2 * h][t]);
              cc += zz[t] * int64_t(R8[k][2 * h + 1][t]);
            }
            const int64_t us = rsr(cs * lut, sh), uc = rsr(cc * lut, sh);
            ph[NB * F + (n * F + fl) * NH + h] = sat(rsr((us * us + uc * uc) * KE, SHE), 8);
          }
        }
      int64_t s = BINT;
      for (int d = 0; d < D; ++d) s += int64_t(A8[d]) * ph[d];
      score[(size_t)b * K + k] = s;
    }
    int64_t best = 0;
    for (int k = 1; k < K; ++k)
      if (score[(size_t)b * K + k] > score[(size_t)b * K + best]) best = k;
    pred[b] = best;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "dung: %s <thu_muc_vector> <B> [so_lan_do]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int B = std::atoi(argv[2]);
  const int TT = S0 + T;
  std::vector<int16_t> in((size_t)B * C * TT);
  FILE *f = std::fopen((dir + "/input.bin").c_str(), "rb");
  if (!f || std::fread(in.data(), 2, in.size(), f) != in.size()) { std::fprintf(stderr, "loi doc input\n"); return 2; }
  std::fclose(f);

  Out o;
  decode(in.data(), B, o);
  dump(dir + "/out_x8.bin", o.x8);
  dump(dir + "/out_z.bin", o.z);
  dump(dir + "/out_N.bin", o.N);
  dump(dir + "/out_phi.bin", o.phi);
  dump(dir + "/out_score.bin", o.score);
  dump(dir + "/out_pred.bin", o.pred);
  std::printf("xong: B=%d T=%d\n", B, T);

  if (argc >= 4) {  // do tre: moi lan goi decode 1 trial (B=1), lan luot qua B trial
    const int R = std::atoi(argv[3]);
    Out o1;
    for (int b = 0; b < B; ++b) decode(&in[(size_t)b * C * TT], 1, o1);  // lam nong
    std::vector<double> us;
    us.reserve((size_t)R);
    int64_t sai = 0;
    for (int r = 0; r < R; ++r) {
      const int b = r % B;
      const auto t0 = std::chrono::steady_clock::now();
      decode(&in[(size_t)b * C * TT], 1, o1);
      const auto t1 = std::chrono::steady_clock::now();
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      sai += (o1.pred[0] != o.pred[b]);
    }
    std::vector<double> s = us;
    std::sort(s.begin(), s.end());
    double tb = 0;
    for (double v : us) tb += v;
    tb /= R;
    auto pct = [&](double p) { return s[std::min<size_t>(s.size() - 1, (size_t)(p * (s.size() - 1) + 0.5))]; };
    std::printf("BENCH T=%d Ttot=%d R=%d mean_us=%.2f median_us=%.2f p95_us=%.2f min_us=%.2f max_us=%.2f pred_lech=%lld\n",
                T, TT, R, tb, pct(0.5), pct(0.95), s.front(), s.back(), (long long)sai);
    FILE *g = std::fopen((dir + "/bench_us.bin").c_str(), "wb");
    std::fwrite(us.data(), sizeof(double), us.size(), g);
    std::fclose(g);
    if (sai) return 4;
  }
  return 0;
}
