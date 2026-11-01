// tt-CCA + bank loc + tru TB (ttcca_fb5cd) va FBCCA fb5cd tren PS ZCU104. C++ float64 thuan, 1 luong.
// Mot quyet dinh: tru TB [0, onset) -> 5 bo loc nhan qua lfilter(b, a) (DF2T, trang thai dau 0) tren [0, s0 + N) -> cat cua so
// -> moi dai: QR Householder cua X^T (mot lan) -> moi muc tieu: M = Qx^T Qy_k, tri rieng lon nhat cua M M^T (Jacobi), w_x = Rx^-1 u,
// rho1, rho2, rho3 -> tong theo dai -> arg max. Tinh truoc (khong tinh vao do tre): Qy_k, X_bar_k da tru TB, w_bar_k^T X_bar_k.
// Dung: ttcca <dir T> check | bench tt|fb R | loop tt|fb giay
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string D;
static long long MACS = 0;
static bool COUNT = false;
#define CNT(n) do { if (COUNT) MACS += (long long)(n); } while (0)

template <typename T> static std::vector<T> rd(const std::string &f, size_t n) {
  std::vector<T> v(n);
  FILE *p = std::fopen((D + "/" + f).c_str(), "rb");
  if (!p || std::fread(v.data(), sizeof(T), n, p) != n) { std::fprintf(stderr, "LOI doc %s\n", f.c_str()); std::exit(2); }
  std::fclose(p);
  return v;
}
static int jint(const std::string &s, const std::string &k) {
  const size_t p = s.find("\"" + k + "\"");
  return std::atoi(s.c_str() + s.find(':', p) + 1);
}

struct Pack {
  int C, K, NB, NH, N, T0, on, s0, M, J;
  std::vector<std::vector<double>> fb, fa;                 // he so loc moi dai
  std::vector<double> fbw, qy, xb, tw, wxb;                 // qy (K, N, J); xb (NB, K, C, N) da tru TB; tw (NB, K, N); wxb (NB, K, C)
  void load() {
    std::ifstream f(D + "/meta.json"); std::stringstream ss; ss << f.rdbuf(); const std::string s = ss.str();
    C = jint(s, "C"); K = jint(s, "K"); NB = jint(s, "NB"); NH = jint(s, "NH"); N = jint(s, "N"); T0 = jint(s, "T0");
    on = jint(s, "onset"); s0 = jint(s, "s0"); M = jint(s, "M"); J = 2 * NH;
    // filt_len: [[lb, la], ...]
    std::vector<int> fl; size_t p = s.find("\"filt_len\""); p = s.find('[', p) + 1;
    for (int i = 0; i < 2 * NB; ++i) { p = s.find_first_of("0123456789", p); fl.push_back(std::atoi(s.c_str() + p)); p = s.find_first_not_of("0123456789", p); }
    size_t tot = 0; for (int x : fl) tot += x;
    auto fv = rd<double>("filt.f64", tot); size_t q = 0;
    for (int j = 0; j < NB; ++j) {
      fb.emplace_back(fv.begin() + q, fv.begin() + q + fl[2 * j]); q += fl[2 * j];
      fa.emplace_back(fv.begin() + q, fv.begin() + q + fl[2 * j + 1]); q += fl[2 * j + 1];
    }
    fbw = rd<double>("fbw.f64", NB);
    auto refs = rd<double>("refs.f64", (size_t)K * J * N);
    xb = rd<double>("xbar.f64", (size_t)NB * K * C * N);
    wxb = rd<double>("wxbar.f64", (size_t)NB * K * C);
    // Qy_k: QR thin cua tham chieu da tru TB (N x J)
    qy.assign((size_t)K * N * J, 0.0);
    std::vector<double> A((size_t)N * J), R((size_t)J * J);
    for (int k = 0; k < K; ++k) {
      for (int jj = 0; jj < J; ++jj) {
        double m = 0; for (int n = 0; n < N; ++n) m += refs[((size_t)k * J + jj) * N + n]; m /= N;
        for (int n = 0; n < N; ++n) A[(size_t)n * J + jj] = refs[((size_t)k * J + jj) * N + n] - m;
      }
      qr(A.data(), N, J, &qy[(size_t)k * N * J], R.data());
    }
    // template tru TB va w_bar^T X_bar
    tw.assign((size_t)NB * K * N, 0.0);
    for (int j = 0; j < NB; ++j) for (int k = 0; k < K; ++k) {
      double *x = &xb[((size_t)j * K + k) * C * N];
      for (int c = 0; c < C; ++c) { double m = 0; for (int n = 0; n < N; ++n) m += x[c * N + n]; m /= N; for (int n = 0; n < N; ++n) x[c * N + n] -= m; }
      const double *w = &wxb[((size_t)j * K + k) * C];
      for (int n = 0; n < N; ++n) { double v = 0; for (int c = 0; c < C; ++c) v += w[c] * x[c * N + n]; tw[((size_t)j * K + k) * N + n] = v; }
    }
  }
  // QR Householder: A (m x n, hang chinh, m >= n) -> Q thin (m x n), R (n x n). A bi ghi de.
  static void qr(double *A, int m, int n, double *Q, double *R) {
    std::vector<double> V((size_t)m * n, 0.0), beta(n, 0.0);
    for (int j = 0; j < n; ++j) {
      double nrm = 0; for (int i = j; i < m; ++i) nrm += A[(size_t)i * n + j] * A[(size_t)i * n + j];
      nrm = std::sqrt(nrm); CNT(m - j);
      const double a0 = A[(size_t)j * n + j], al = a0 >= 0 ? -nrm : nrm;
      double vn = 0;
      for (int i = j; i < m; ++i) { const double v = A[(size_t)i * n + j] - (i == j ? al : 0); V[(size_t)i * n + j] = v; vn += v * v; }
      beta[j] = vn > 0 ? 2.0 / vn : 0.0;
      for (int c = j; c < n; ++c) {
        double s = 0; for (int i = j; i < m; ++i) s += V[(size_t)i * n + j] * A[(size_t)i * n + c];
        s *= beta[j]; for (int i = j; i < m; ++i) A[(size_t)i * n + c] -= s * V[(size_t)i * n + j];
        CNT(2 * (m - j));
      }
    }
    for (int i = 0; i < n; ++i) for (int c = 0; c < n; ++c) R[i * n + c] = c >= i ? A[(size_t)i * n + c] : 0.0;
    for (int i = 0; i < m; ++i) for (int c = 0; c < n; ++c) Q[(size_t)i * n + c] = (i == c) ? 1.0 : 0.0;
    for (int j = n - 1; j >= 0; --j)
      for (int c = 0; c < n; ++c) {
        double s = 0; for (int i = j; i < m; ++i) s += V[(size_t)i * n + j] * Q[(size_t)i * n + c];
        s *= beta[j]; for (int i = j; i < m; ++i) Q[(size_t)i * n + c] -= s * V[(size_t)i * n + j];
        CNT(2 * (m - j));
      }
  }
};

// tri rieng lon nhat va vector rieng cua ma tran doi xung S (n x n), Jacobi vong
static double top_eig(std::vector<double> S, int n, double *u) {
  std::vector<double> V((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0, tot = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { tot += S[i * n + j] * S[i * n + j]; if (i != j) off += S[i * n + j] * S[i * n + j]; }
    if (off <= 1e-30 * tot) break;
    for (int p = 0; p < n - 1; ++p)
      for (int q = p + 1; q < n; ++q) {
        const double apq = S[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double th = (S[q * n + q] - S[p * n + p]) / (2 * apq);
        const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; ++k) {
          const double a = S[k * n + p], b = S[k * n + q];
          S[k * n + p] = c * a - s * b; S[k * n + q] = s * a + c * b;
        }
        for (int k = 0; k < n; ++k) {
          const double a = S[p * n + k], b = S[q * n + k];
          S[p * n + k] = c * a - s * b; S[q * n + k] = s * a + c * b;
        }
        for (int k = 0; k < n; ++k) {
          const double a = V[k * n + p], b = V[k * n + q];
          V[k * n + p] = c * a - s * b; V[k * n + q] = s * a + c * b;
        }
        CNT(6 * n);
      }
  }
  int b = 0; for (int i = 1; i < n; ++i) if (S[i * n + i] > S[b * n + b]) b = i;
  for (int i = 0; i < n; ++i) u[i] = V[i * n + b];
  return S[b * n + b];
}

static double pearson_c(const double *a, const double *b, int n) {     // a, b da tru TB
  double ab = 0, aa = 0, bb = 0;
  for (int i = 0; i < n; ++i) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  CNT(3 * n);
  const double d = std::sqrt(aa * bb);
  return d > 0 ? ab / d : 0.0;
}

struct Times { double filt = 0, corr = 0; };
static std::vector<double> XF;                                          // (NB, C, N) sau loc, de doi chieu cong 1

// tra ve du doan; score (K); tt = true: tt-CCA, false: FBCCA
static int decide(const Pack &P, const double *raw, bool tt, std::vector<double> &score, Times *tm = nullptr) {
  const int C = P.C, N = P.N, T0 = P.T0, K = P.K, J = P.J;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<double> seg((size_t)C * T0), y(T0);
  for (int c = 0; c < C; ++c) {
    double m = 0; for (int t = 0; t < P.on; ++t) m += raw[(size_t)c * T0 + t]; m /= P.on;
    for (int t = 0; t < T0; ++t) seg[(size_t)c * T0 + t] = raw[(size_t)c * T0 + t] - m;
  }
  XF.assign((size_t)P.NB * C * N, 0.0);
  for (int j = 0; j < P.NB; ++j) {
    const auto &b = P.fb[j], &a = P.fa[j];
    const int L = (int)std::max(b.size(), a.size());
    std::vector<double> bb(L, 0.0), aa(L, 0.0), z(L, 0.0);
    for (size_t i = 0; i < b.size(); ++i) bb[i] = b[i] / a[0];
    for (size_t i = 0; i < a.size(); ++i) aa[i] = a[i] / a[0];
    for (int c = 0; c < C; ++c) {
      std::fill(z.begin(), z.end(), 0.0);
      const double *x = &seg[(size_t)c * T0];
      for (int t = 0; t < T0; ++t) {                                    // DF2T nhu scipy.signal.lfilter
        const double yt = bb[0] * x[t] + z[0];
        for (int i = 1; i < L - 1; ++i) z[i - 1] = bb[i] * x[t] + z[i] - aa[i] * yt;
        z[L - 2] = bb[L - 1] * x[t] - aa[L - 1] * yt;
        if (t >= P.s0) XF[((size_t)j * C + c) * N + (t - P.s0)] = yt;
      }
      CNT((long long)T0 * 2 * L);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  score.assign(K, 0.0);
  std::vector<double> X((size_t)C * N), A((size_t)N * C), Qx((size_t)N * C), Rx((size_t)C * C), Mm((size_t)C * J), S((size_t)C * C);
  std::vector<double> u(C), wx(C), p1(N), p2(N), p3(N);
  for (int j = 0; j < P.NB; ++j) {
    for (int c = 0; c < C; ++c) {
      const double *x = &XF[((size_t)j * C + c) * N];
      double m = 0; for (int n = 0; n < N; ++n) m += x[n]; m /= N;
      for (int n = 0; n < N; ++n) { X[(size_t)c * N + n] = x[n] - m; A[(size_t)n * C + c] = x[n] - m; }
    }
    Pack::qr(A.data(), N, C, Qx.data(), Rx.data());
    for (int k = 0; k < K; ++k) {
      const double *Qy = &P.qy[(size_t)k * N * J];
      std::fill(Mm.begin(), Mm.end(), 0.0);
      for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c) { const double q = Qx[(size_t)n * C + c]; for (int jj = 0; jj < J; ++jj) Mm[c * J + jj] += q * Qy[(size_t)n * J + jj]; }
      CNT((long long)N * C * J);
      for (int a = 0; a < C; ++a) for (int b = a; b < C; ++b) { double s = 0; for (int jj = 0; jj < J; ++jj) s += Mm[a * J + jj] * Mm[b * J + jj]; S[a * C + b] = S[b * C + a] = s; }
      CNT(C * (C + 1) / 2 * J);
      const double lam = top_eig(S, C, u.data());
      const double rho2 = std::min(std::sqrt(std::max(lam, 0.0)), 1.0);
      if (!tt) { score[k] += P.fbw[j] * rho2 * rho2; continue; }
      for (int c = C - 1; c >= 0; --c) { double s = u[c]; for (int q = c + 1; q < C; ++q) s -= Rx[c * C + q] * wx[q]; wx[c] = s / Rx[c * C + c]; }
      CNT(C * C / 2);
      const double *xb = &P.xb[((size_t)j * K + k) * C * N], *w = &P.wxb[((size_t)j * K + k) * C], *tw = &P.tw[((size_t)j * K + k) * N];
      for (int n = 0; n < N; ++n) {
        double a1 = 0, a2 = 0, a3 = 0;
        for (int c = 0; c < C; ++c) { a1 += w[c] * X[(size_t)c * N + n]; a2 += wx[c] * xb[(size_t)c * N + n]; a3 += wx[c] * X[(size_t)c * N + n]; }
        p1[n] = a1; p2[n] = a2; p3[n] = a3;
      }
      CNT((long long)3 * N * C);
      score[k] += P.fbw[j] * (pearson_c(tw, p1.data(), N) + rho2 + pearson_c(p2.data(), p3.data(), N));
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  if (tm) { tm->filt += std::chrono::duration<double, std::micro>(t1 - t0).count(); tm->corr += std::chrono::duration<double, std::micro>(t2 - t1).count(); }
  return (int)(std::max_element(score.begin(), score.end()) - score.begin());
}

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "dung: %s <dir T> check | bench tt|fb R | loop tt|fb giay\n", argv[0]); return 2; }
  D = argv[1];
  const std::string mode = argv[2];
  Pack P; P.load();
  const int M = P.M, C = P.C, K = P.K, N = P.N, T0 = P.T0, NB = P.NB;
  auto raw = rd<double>("raw.f64", (size_t)M * C * T0);
  std::vector<double> sc;
  if (mode == "check") {
    auto rx = rd<double>("ref_xfilt.f64", (size_t)M * NB * C * N);
    auto rt = rd<double>("ref_tt.f64", (size_t)M * K), rf = rd<double>("ref_fb.f64", (size_t)M * K);
    auto lab = rd<int32_t>("label.i32", M);
    double fmax = 0, fref = 0, dtt = 0, dfb = 0;
    long same[2] = {0, 0}, tie[2] = {0, 0}, bad[2] = {0, 0}, corr[2] = {0, 0}; long long mac[2] = {0, 0};
    for (int m = 0; m < M; ++m)
      for (int v = 0; v < 2; ++v) {
        COUNT = (m == 0); MACS = 0;
        const int out = decide(P, &raw[(size_t)m * C * T0], v == 0, sc);
        if (m == 0) mac[v] = MACS;
        COUNT = false;
        if (v == 0) for (size_t q = 0; q < (size_t)NB * C * N; ++q) {
          fmax = std::max(fmax, std::fabs(XF[q] - rx[(size_t)m * NB * C * N + q])); fref = std::max(fref, std::fabs(rx[(size_t)m * NB * C * N + q]));
        }
        const double *ref = v == 0 ? &rt[(size_t)m * K] : &rf[(size_t)m * K];
        double &dm = v == 0 ? dtt : dfb;
        for (int k = 0; k < K; ++k) dm = std::max(dm, std::fabs(sc[k] - ref[k]));
        const int rp = (int)(std::max_element(ref, ref + K) - ref);
        corr[v] += out == lab[m];
        if (out == rp) ++same[v];
        else {
          std::vector<double> s2(ref, ref + K); std::sort(s2.begin(), s2.end());
          const double marg = s2[K - 1] - s2[K - 2];
          if (marg < 1e-9) ++tie[v]; else ++bad[v];
          std::printf("%s luot %d: lech, le %.3g\n", v == 0 ? "tt" : "fb", m, marg);
        }
      }
    const double frel = fmax / fref;
    const bool g1 = frel <= 1e-9, g2 = dtt <= 1e-6 && dfb <= 1e-6, g3 = bad[0] == 0 && bad[1] == 0;
    std::printf("T=%d M=%d C=%d K=%d NB=%d\n", N, M, C, K, NB);
    std::printf("CONG 1 loc: sai so tuong doi lon nhat %.3g (nguong 1e-9): %s\n", frel, g1 ? "DAT" : "KHONG DAT");
    std::printf("CONG 2 diem: |d| lon nhat tt %.3g, fb %.3g (nguong 1e-6): %s\n", dtt, dfb, g2 ? "DAT" : "KHONG DAT");
    std::printf("CONG 3 du doan trung tt %ld/%d, fb %ld/%d; hoa so hoc %ld/%ld; lech that %ld/%ld: %s\n", same[0], M, same[1], M,
                tie[0], tie[1], bad[0], bad[1], g3 ? "DAT" : "KHONG DAT");
    std::printf("mo ta: acc tt %.4f fb %.4f; MAC dem duoc mot quyet dinh: tt %lld fb %lld\n", double(corr[0]) / M, double(corr[1]) / M,
                mac[0], mac[1]);
    return (g1 && g2 && g3) ? 0 : 3;
  }
  const bool tt = argc > 3 && std::string(argv[3]) == "tt";
  if (mode == "bench") {
    const int R = argc > 4 ? std::atoi(argv[4]) : 200;
    for (int i = 0; i < 8; ++i) decide(P, &raw[(size_t)i * C * T0], tt, sc);
    std::vector<double> us(R); Times tm;
    for (int r = 0; r < R; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      decide(P, &raw[(size_t)(r % 8) * C * T0], tt, sc, &tm);
      us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    std::vector<double> s = us; std::sort(s.begin(), s.end());
    auto pct = [&](double p) { return s[std::min<size_t>(s.size() - 1, (size_t)(p * (s.size() - 1) + 0.5))]; };
    std::printf("BENCH m=%s T=%d R=%d median_us=%.1f p95_us=%.1f min_us=%.1f max_us=%.1f mean_filt_us=%.1f mean_corr_us=%.1f\n",
                tt ? "tt" : "fb", N, R, pct(0.5), pct(0.95), s.front(), s.back(), tm.filt / R, tm.corr / R);
    FILE *g = std::fopen((D + "/bench_us_" + (tt ? "tt" : "fb") + ".bin").c_str(), "wb");
    std::fwrite(us.data(), sizeof(double), us.size(), g); std::fclose(g);
    return 0;
  }
  if (mode == "loop") {
    const double dur = argc > 4 ? std::atof(argv[4]) : 60;
    long n = 0; const auto t0 = std::chrono::steady_clock::now(); double el = 0;
    while (el < dur) { decide(P, &raw[(size_t)(n % 8) * C * T0], tt, sc); ++n;
      el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    std::printf("LOOP m=%s T=%d n=%ld elapsed=%.3f\n", tt ? "tt" : "fb", N, n, el);
    return 0;
  }
  return 2;
}
