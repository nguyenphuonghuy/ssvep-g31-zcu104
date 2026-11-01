// Ensemble-DNN (Guney & Ozkan 2023) — suy luan 1 quyet dinh tren CPU, 1 luong.
// Phep tinh chep tu scripts/ensemble_dnn.py (PyTorch) + scipy.signal.sosfiltfilt (padtype odd) + testing.m (chon dong k).
// DNN float32 (nhu PyTorch); loc va do tuong dong double (nhu ban PyTorch dung .double()).
// Phan tinh truoc duoc tu template (ket hop dai con, chieu 120 cot, tru TB, chuan) lam LUC NAP, khong tinh vao do tre.
// Bien dich: g++ -std=c++17 -O2 ensdnn.cpp -o ensdnn        (ban A; ban B: -O3 -mcpu=cortex-a53)
// Dung:  ./ensdnn <dir T> check            -> cong 1-3 so voi chuan PyTorch tren M luot thu (in "CONG ... DAT/KHONG DAT")
//        ./ensdnn <dir T> bench <R>        -> 1 luot do: lam nong 8, R quyet dinh xoay vong 8 luot thu; in BENCH ...
//        ./ensdnn <dir T> loop <giay>      -> giai ma lien tuc trong <giay> (do cong suat); in LOOP n=... elapsed=...
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static constexpr int NCH = 120, KW = 10, PADL = 4;     // padding "same" cua PyTorch voi kernel 10: trai 4, phai 5

struct Man { int N, C, K, T, M, NH, padlen, per_model; };

static std::string D;
template <typename Tp> static std::vector<Tp> rd(const std::string &name, size_t n) {
  std::vector<Tp> v(n);
  FILE *f = std::fopen((D + "/" + name).c_str(), "rb");
  if (!f || std::fread(v.data(), sizeof(Tp), n, f) != n) { std::fprintf(stderr, "LOI doc %s\n", name.c_str()); std::exit(2); }
  std::fclose(f);
  return v;
}
static int jint(const std::string &js, const char *key) {
  size_t p = js.find(std::string("\"") + key + "\"");
  if (p == std::string::npos) { std::fprintf(stderr, "manifest thieu %s\n", key); std::exit(2); }
  return std::atoi(js.c_str() + js.find(':', p) + 1);
}

struct Model { const float *sub, *chw, *chb, *dw, *db, *cw, *cb, *fw, *fb; };

struct Ens {
  Man m{};
  std::vector<float> W;          // trong so
  std::vector<Model> mod;
  std::vector<double> sos, zi, Q;  // sos (3,6), zi (3,2), Q (K,T,R)
  std::vector<float> bc;         // template da chieu, tru TB: (N, K, 120, T)
  std::vector<double> bb;        // |bc|^2: (N, K, 120)
  std::vector<double> Wd;        // Wc double: (N, 120, C)
  std::vector<double> wsd;       // trong so dai con double: (N, 3)
  // bo dem tam
  std::vector<float> xf, s, h1, h2, h3;
  std::vector<double> ext, y1, xs, a;

  void load() {
    std::string js;
    { FILE *f = std::fopen((D + "/manifest.json").c_str(), "rb"); char b[4096]; size_t n = std::fread(b, 1, sizeof b, f);
      js.assign(b, n); std::fclose(f); }
    m = {jint(js, "N"), jint(js, "C"), jint(js, "K"), jint(js, "T"), jint(js, "M"), jint(js, "NH"), jint(js, "padlen"),
         jint(js, "per_model")};
    const int C = m.C, T = m.T, K = m.K, T2 = T / 2, N = m.N, R = 2 * m.NH;
    const int per = 3 + NCH * C + NCH + NCH * NCH * 2 + NCH + NCH * NCH * KW + NCH + K * NCH * T2 + K;
    if (per != m.per_model) { std::fprintf(stderr, "so tham so/mo hinh %d != manifest %d\n", per, m.per_model); std::exit(2); }
    W = rd<float>("weights.f32", (size_t)N * per);
    for (int n = 0; n < N; ++n) {
      const float *p = &W[(size_t)n * per];
      Model q;
      q.sub = p; p += 3; q.chw = p; p += NCH * C; q.chb = p; p += NCH; q.dw = p; p += NCH * NCH * 2; q.db = p; p += NCH;
      q.cw = p; p += NCH * NCH * KW; q.cb = p; p += NCH; q.fw = p; p += (size_t)K * NCH * T2; q.fb = p;
      mod.push_back(q);
    }
    sos = rd<double>("sos.f64", 18); zi = rd<double>("zi.f64", 6); Q = rd<double>("Q.f64", (size_t)K * T * R);
    // ---- tinh truoc tu template (khong tinh vao do tre) ----
    std::vector<float> tm = rd<float>("tmpl.f32", (size_t)N * K * 3 * C * T);
    bc.assign((size_t)N * K * NCH * T, 0.f); bb.assign((size_t)N * K * NCH, 0.0);
    Wd.assign((size_t)N * NCH * C, 0.0); wsd.assign((size_t)N * 3, 0.0);
    std::vector<double> tk((size_t)C * T), bj(T);
    for (int n = 0; n < N; ++n) {
      for (int b = 0; b < 3; ++b) wsd[n * 3 + b] = mod[n].sub[b];
      for (int j = 0; j < NCH; ++j) for (int c = 0; c < C; ++c) Wd[((size_t)n * NCH + j) * C + c] = mod[n].chw[j * C + c];
      for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t) {
          double v = 0; for (int b = 0; b < 3; ++b) v += wsd[n * 3 + b] * (double)tm[((((size_t)n * K + k) * 3 + b) * C + c) * T + t];
          tk[(size_t)c * T + t] = v;
        }
        for (int j = 0; j < NCH; ++j) {
          double mu = 0;
          for (int t = 0; t < T; ++t) { double v = 0; for (int c = 0; c < C; ++c) v += Wd[((size_t)n * NCH + j) * C + c] * tk[(size_t)c * T + t]; bj[t] = v; mu += v; }
          mu /= T; double ss = 0;
          for (int t = 0; t < T; ++t) { double v = bj[t] - mu; bc[(((size_t)n * K + k) * NCH + j) * T + t] = (float)v; ss += v * v; }
          bb[((size_t)n * K + k) * NCH + j] = ss;
        }
      }
    }
    xf.resize((size_t)3 * C * T); s.resize((size_t)C * T); h1.resize((size_t)NCH * T); h2.resize((size_t)NCH * T2);
    h3.resize((size_t)NCH * T2); ext.resize(T + 2 * m.padlen); y1.resize(T + 2 * m.padlen); xs.resize((size_t)C * T); a.resize(T);
  }

  // scipy.signal.sosfiltfilt(sos, x, padlen=P) voi padtype 'odd', 1 khau bac hai (DF-II chuyen vi nhu sosfilt)
  void filtfilt(int band, const double *x, double *y) {
    const int T = m.T, P = m.padlen, L = T + 2 * P;
    const double *q = &sos[band * 6]; const double b0 = q[0], b1 = q[1], b2 = q[2], a1 = q[4] / q[3], a2 = q[5] / q[3];
    const double z0i = zi[band * 2], z1i = zi[band * 2 + 1];
    for (int i = 0; i < P; ++i) ext[i] = 2 * x[0] - x[P - i];
    for (int t = 0; t < T; ++t) ext[P + t] = x[t];
    for (int i = 0; i < P; ++i) ext[P + T + i] = 2 * x[T - 1] - x[T - 2 - i];
    double z0 = z0i * ext[0], z1 = z1i * ext[0];
    for (int i = 0; i < L; ++i) { const double xi = ext[i], yi = b0 * xi + z0; z0 = b1 * xi - a1 * yi + z1; z1 = b2 * xi - a2 * yi; y1[i] = yi; }
    z0 = z0i * y1[L - 1]; z1 = z1i * y1[L - 1];
    for (int i = L - 1; i >= 0; --i) { const double xi = y1[i], yi = b0 * xi + z0; z0 = b1 * xi - a1 * yi + z1; z1 = b2 * xi - a2 * yi; ext[i] = yi; }
    for (int t = 0; t < T; ++t) y[t] = ext[P + t];
  }

  // loc 3 dai cho mot luot thu tho (C,T) -> xf (3,C,T) float32
  void prep(const double *raw) {
    const int C = m.C, T = m.T;
    std::vector<double> yb(T);
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < C; ++c) { filtfilt(b, raw + (size_t)c * T, yb.data()); for (int t = 0; t < T; ++t) xf[((size_t)b * C + c) * T + t] = (float)yb[t]; }
  }

  // GuneyDNN.forward (eval) cho mo hinh n tren xf -> logit[K]
  void forward(int n, float *logit) {
    const Model &q = mod[n]; const int C = m.C, T = m.T, T2 = T / 2, K = m.K;
    for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t)
      s[(size_t)c * T + t] = q.sub[0] * xf[(size_t)c * T + t] + q.sub[1] * xf[((size_t)C + c) * T + t] + q.sub[2] * xf[((size_t)2 * C + c) * T + t];
    for (int j = 0; j < NCH; ++j) {
      float *o = &h1[(size_t)j * T];
      for (int t = 0; t < T; ++t) o[t] = q.chb[j];
      for (int c = 0; c < C; ++c) { const float w = q.chw[j * C + c]; const float *sc = &s[(size_t)c * T]; for (int t = 0; t < T; ++t) o[t] += w * sc[t]; }
    }
    for (int oc = 0; oc < NCH; ++oc) {
      float *o = &h2[(size_t)oc * T2];
      for (int u = 0; u < T2; ++u) o[u] = q.db[oc];
      for (int ic = 0; ic < NCH; ++ic) {
        const float w0 = q.dw[((size_t)oc * NCH + ic) * 2], w1 = q.dw[((size_t)oc * NCH + ic) * 2 + 1]; const float *in = &h1[(size_t)ic * T];
        for (int u = 0; u < T2; ++u) o[u] += w0 * in[2 * u] + w1 * in[2 * u + 1];
      }
      for (int u = 0; u < T2; ++u) o[u] = o[u] > 0.f ? o[u] : 0.f;     // ReLU
    }
    for (int oc = 0; oc < NCH; ++oc) {
      float *o = &h3[(size_t)oc * T2];
      for (int u = 0; u < T2; ++u) o[u] = q.cb[oc];
      for (int ic = 0; ic < NCH; ++ic) {
        const float *w = &q.cw[((size_t)oc * NCH + ic) * KW]; const float *in = &h2[(size_t)ic * T2];
        for (int k = 0; k < KW; ++k) {
          const int sh = k - PADL; const int u0 = std::max(0, -sh), u1 = std::min(T2, T2 - sh); const float wk = w[k];
          for (int u = u0; u < u1; ++u) o[u] += wk * in[u + sh];
        }
      }
    }
    const int F = NCH * T2;
    for (int k = 0; k < K; ++k) { const float *w = &q.fw[(size_t)k * F]; float v = q.fb[k]; for (int i = 0; i < F; ++i) v += w[i] * h3[i]; logit[k] = v; }
  }

  // do tuong dong rho_n cho mo hinh n, lop du doan p (dung xf)
  double similarity(int n, int p) {
    const int C = m.C, T = m.T, R = 2 * m.NH;
    for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t) {
      double v = 0; for (int b = 0; b < 3; ++b) v += wsd[n * 3 + b] * (double)xf[((size_t)b * C + c) * T + t];
      xs[(size_t)c * T + t] = v;
    }
    const double *Qp = &Q[(size_t)p * T * R];
    double best = -INFINITY;
    for (int j = 0; j < NCH; ++j) {
      double mu = 0;
      for (int t = 0; t < T; ++t) { double v = 0; for (int c = 0; c < C; ++c) v += Wd[((size_t)n * NCH + j) * C + c] * xs[(size_t)c * T + t]; a[t] = v; mu += v; }
      mu /= T; double saa = 0, sab = 0;
      const float *bj = &bc[(((size_t)n * m.K + p) * NCH + j) * T];
      for (int t = 0; t < T; ++t) { a[t] -= mu; saa += a[t] * a[t]; sab += a[t] * (double)bj[t]; }
      double cc = 0;
      for (int r = 0; r < R; ++r) { double v = 0; for (int t = 0; t < T; ++t) v += a[t] * Qp[(size_t)t * R + r]; cc += v * v; }
      const double s_ = sab * sab / (saa * bb[((size_t)n * m.K + p) * NCH + j]) + cc / saa;
      if (!std::isnan(s_) && s_ > best) best = s_;
    }
    return best;
  }
};

// chon dong k (testing.m): sap giam dan on dinh; phieu tich luy; do tin = top1 - top2; k* = argmax dau tien
static int select(const std::vector<double> &sims, const std::vector<int> &preds, int K, int &kstar, double &margin_vote,
                  double &margin_conf) {
  const int N = (int)sims.size();
  std::vector<int> ord(N); std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int i, int j) { return sims[i] > sims[j]; });
  std::vector<double> V((size_t)N * K, 0.0), cur(K, 0.0), conf(N);
  for (int k = 0; k < N; ++k) {
    cur[preds[ord[k]]] += sims[ord[k]];
    std::copy(cur.begin(), cur.end(), V.begin() + (size_t)k * K);
    std::vector<double> t(cur); std::partial_sort(t.begin(), t.begin() + 2, t.end(), std::greater<double>());
    conf[k] = t[0] - t[1];
  }
  int kb = 0; for (int k = 1; k < N; ++k) if (conf[k] > conf[kb]) kb = k;
  double c2 = -INFINITY; for (int k = 0; k < N; ++k) if (k != kb) c2 = std::max(c2, conf[k]);
  margin_conf = conf[kb] - c2;
  const double *v = &V[(size_t)kb * K];
  int p = 0; for (int i = 1; i < K; ++i) if (v[i] > v[p]) p = i;
  double v2 = -INFINITY; for (int i = 0; i < K; ++i) if (i != p) v2 = std::max(v2, v[i]);
  margin_vote = v[p] - v2;
  kstar = kb + 1;
  return p;
}

struct Times { double filt = 0, dnn = 0, sim = 0; };

static int decide(Ens &E, const double *raw, std::vector<float> &logits, std::vector<int> &preds, std::vector<double> &sims,
                  int &kstar, double &mv, double &mc, Times *tt = nullptr) {
  using clk = std::chrono::steady_clock;
  const int N = E.m.N, K = E.m.K;
  auto t0 = clk::now();
  E.prep(raw);
  auto t1 = clk::now();
  for (int n = 0; n < N; ++n) {
    float *l = &logits[(size_t)n * K]; E.forward(n, l);
    int p = 0; for (int k = 1; k < K; ++k) if (l[k] > l[p]) p = k;
    preds[n] = p;
  }
  auto t2 = clk::now();
  for (int n = 0; n < N; ++n) sims[n] = E.similarity(n, preds[n]);
  const int out = select(sims, preds, K, kstar, mv, mc);
  auto t3 = clk::now();
  if (tt) { tt->filt += std::chrono::duration<double, std::micro>(t1 - t0).count();
            tt->dnn += std::chrono::duration<double, std::micro>(t2 - t1).count();
            tt->sim += std::chrono::duration<double, std::micro>(t3 - t2).count(); }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "dung: %s <dir T> check|bench R|loop giay\n", argv[0]); return 2; }
  D = argv[1];
  const std::string mode = argv[2];
  Ens E; E.load();
  const int N = E.m.N, C = E.m.C, K = E.m.K, T = E.m.T, M = E.m.M;
  std::vector<double> raw = rd<double>("raw.f64", (size_t)M * C * T);
  std::vector<float> logits((size_t)N * K); std::vector<int> preds(N); std::vector<double> sims(N);
  int kstar; double mv, mc;
  if (mode == "check") {
    // cong 2: chuan PyTorch float32 CPU (sua doi 2026-10-22); chuan luc xuat (vmap GPU) chi mo ta
    auto rx = rd<float>("ref_xfilt.f32", (size_t)M * 3 * C * T); auto rl = rd<float>("ref_logits_cpu.f32", (size_t)N * M * K);
    auto rp = rd<int32_t>("ref_preds_cpu.i32", (size_t)N * M); auto rs = rd<double>("ref_sims.f64", (size_t)N * M);
    auto rlx = rd<float>("ref_logits.f32", (size_t)N * M * K); double lxmax = 0;
    auto rk = rd<int32_t>("ref_kstar.i32", M); auto rf = rd<int32_t>("ref_pred.i32", M); auto lab = rd<int32_t>("label.i32", M);
    double fmax = 0, fref = 0, lmax = 0, smax = 0; long pm = 0, ksame = 0, fsame = 0, correct = 0, tie = 0, bad = 0;
    for (int i = 0; i < M; ++i) {
      const int out = decide(E, &raw[(size_t)i * C * T], logits, preds, sims, kstar, mv, mc);
      for (size_t q = 0; q < (size_t)3 * C * T; ++q) { fmax = std::max(fmax, (double)std::fabs(E.xf[q] - rx[(size_t)i * 3 * C * T + q])); fref = std::max(fref, (double)std::fabs(rx[(size_t)i * 3 * C * T + q])); }
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          lmax = std::max(lmax, (double)std::fabs(logits[(size_t)n * K + k] - rl[((size_t)n * M + i) * K + k]));
          lxmax = std::max(lxmax, (double)std::fabs(logits[(size_t)n * K + k] - rlx[((size_t)n * M + i) * K + k]));
        }
        pm += preds[n] == rp[(size_t)n * M + i];
        smax = std::max(smax, std::fabs(sims[n] - rs[(size_t)n * M + i]));
      }
      ksame += kstar == rk[i]; correct += out == lab[i];
      if (out == rf[i]) ++fsame;
      else if (std::min(mv, mc) < 1e-5) { ++tie; std::printf("luot %d: lech, le quyet dinh %.3g (hoa so hoc)\n", i, std::min(mv, mc)); }
      else { ++bad; std::printf("luot %d: lech, le %.3g (KHONG phai hoa)\n", i, std::min(mv, mc)); }
    }
    const double frel = fmax / fref;
    const bool g1 = frel <= 1e-4, g2 = lmax <= 1e-3 && pm >= (long)std::ceil(0.999 * N * M), g3 = bad == 0;
    std::printf("T=%d N=%d M=%d\n", T, N, M);
    std::printf("CONG 1 loc: sai so tuong doi lon nhat %.3g (nguong 1e-4): %s\n", frel, g1 ? "DAT" : "KHONG DAT");
    std::printf("CONG 2 logit (chuan PyTorch CPU): |d| lon nhat %.3g (nguong 1e-3); du doan tung mang trung %ld/%ld: %s\n", lmax,
                pm, (long)N * M, g2 ? "DAT" : "KHONG DAT");
    std::printf("mo ta: |d logit| voi chuan luc xuat (vmap GPU) %.3g\n", lxmax);
    std::printf("CONG 3 du doan cuoi trung %ld/%d; hoa so hoc %ld; lech that %ld: %s\n", fsame, M, tie, bad, g3 ? "DAT" : "KHONG DAT");
    std::printf("mo ta: |d sims| lon nhat %.3g; k* trung %ld/%d; acc %.4f\n", smax, ksame, M, double(correct) / M);
    return (g1 && g2 && g3) ? 0 : 3;
  }
  if (mode == "bench") {
    const int R = argc > 3 ? std::atoi(argv[3]) : 200;
    for (int i = 0; i < 8; ++i) decide(E, &raw[(size_t)i * C * T], logits, preds, sims, kstar, mv, mc);
    std::vector<double> us(R); Times tt;
    for (int r = 0; r < R; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      decide(E, &raw[(size_t)(r % 8) * C * T], logits, preds, sims, kstar, mv, mc, &tt);
      us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    std::vector<double> s = us; std::sort(s.begin(), s.end());
    auto pct = [&](double p) { return s[std::min<size_t>(s.size() - 1, (size_t)(p * (s.size() - 1) + 0.5))]; };
    std::printf("BENCH T=%d N=%d R=%d median_us=%.1f p95_us=%.1f min_us=%.1f max_us=%.1f mean_filt_us=%.1f mean_dnn_us=%.1f "
                "mean_sim_us=%.1f\n", T, N, R, pct(0.5), pct(0.95), s.front(), s.back(), tt.filt / R, tt.dnn / R, tt.sim / R);
    FILE *g = std::fopen((D + "/bench_us.bin").c_str(), "wb"); std::fwrite(us.data(), sizeof(double), us.size(), g); std::fclose(g);
    return 0;
  }
  if (mode == "loop") {
    const double dur = argc > 3 ? std::atof(argv[3]) : 60;
    long n = 0; const auto t0 = std::chrono::steady_clock::now(); double el = 0;
    while (el < dur) { decide(E, &raw[(size_t)(n % 8) * C * T], logits, preds, sims, kstar, mv, mc); ++n;
      el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
    std::printf("LOOP T=%d n=%ld elapsed=%.3f\n", T, n, el);
    return 0;
  }
  return 2;
}
