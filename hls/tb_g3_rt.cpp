// Testbench kernel v2 (nap tham so khi chay): voi moi fold trong danh sach, nap fold<f>_params.bin (mode 0), giai ma
// B luot thu dau cua fold<f>_input.bin (mode 1), so score/pred voi golden fold<f>_score/pred.bin.
// Dung: tb_rt <thu_muc T> <danh sach fold, vd 1,2,35 hoac all> [B=240] [expected_phi.bin cua fold 1, 8 luot thu dau]
//   -> "TB DAT" va tra 0 khi khop het. Doi so 4 chi dung o C-sim (cosim: RTL khong ghi mang debug phi).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "g3_decode_rt.h"
#ifndef __SYNTHESIS__
extern int64_t g3_dbg_phi[K][D];
#endif
template <typename Tp> static bool rd(const std::string &p, std::vector<Tp> &v) {
  FILE *f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  const size_t n = std::fread(v.data(), sizeof(Tp), v.size(), f);
  std::fclose(f);
  return n == v.size();
}
int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  std::string fl = argc > 2 ? argv[2] : "1";
  const int B = argc > 3 ? std::atoi(argv[3]) : 240;
  std::vector<int64_t> ephi;
  if (argc > 4) {
    ephi.resize((size_t)8 * K * D);
    if (!rd(argv[4], ephi)) { std::printf("TB LOI doc %s\n", argv[4]); return 2; }
  }
  long sai_phi = 0, so_phi = 0;
  if (fl == "all") { fl.clear(); for (int f = 1; f <= 35; ++f) fl += std::to_string(f) + (f < 35 ? "," : ""); }
  const int TT = S0 + T;
  long sai_s = 0, sai_p = 0, dung = 0, tong = 0, nfold = 0, loi_nap = 0;
  size_t pos = 0;
  while (pos < fl.size()) {
    size_t e = fl.find(',', pos);
    if (e == std::string::npos) e = fl.size();
    const int fold = std::atoi(fl.substr(pos, e - pos).c_str());
    pos = e + 1;
    const std::string pre = dir + "/fold" + std::to_string(fold) + "_";
    std::vector<int32_t> prm(PRM_LEN);
    std::vector<int16_t> in((size_t)B * C * TT);   // theo B (BETA 160 luot)
    std::vector<int64_t> es((size_t)B * K), ep(B), lab(B);
    if (!rd(pre + "params.bin", prm) || !rd(pre + "input.bin", in) || !rd(pre + "score.bin", es) ||
        !rd(pre + "pred.bin", ep) || !rd(pre + "label.bin", lab)) {
      std::printf("TB LOI doc fold %d\n", fold);
      return 2;
    }
    int64_t score[K];
    int32_t pred[1], status[1];
    std::vector<int16_t> dummy((size_t)C * TT, 0);
    g3_decode_rt(dummy.data(), prm.data(), score, pred, status, 0);
    {  // checksum doc nguoc (v2.1): 5 doan [0, OFF_W8), W8, A8, P8, R8; cs = sum v * ((i % 1021) + 1)
      const int b0 = 9 + NB + NB * MAXSEC * 6, b1 = b0 + F * C, b2 = b1 + D, b3 = b2 + NB * K * F * T, b4 = PRM_LEN;
      const int lo[5] = {0, b0, b1, b2, b3}, hi[5] = {b0, b1, b2, b3, b4};
      for (int g = 0; g < 5; ++g) {
        int64_t cs = 0;
        for (int i = lo[g]; i < hi[g]; ++i) cs += int64_t(g == 0 ? prm[i] : int32_t(int8_t(prm[i]))) * ((i % 1021) + 1);
        if (cs != score[g]) { ++loi_nap; std::printf("fold %d: checksum doan %d SAI (hw %lld, ky vong %lld)\n", fold, g,
                                                     (long long)score[g], (long long)cs); }
      }
    }
    if (status[0] != 1) { ++loi_nap; std::printf("fold %d: nap tham so LOI status=%d\n", fold, status[0]); continue; }
    long ss = 0, sp = 0, d = 0;
    for (int b = 0; b < B; ++b) {
      g3_decode_rt(&in[(size_t)b * C * TT], prm.data(), score, pred, status, 1);
      for (int k = 0; k < K; ++k) ss += score[k] != es[(size_t)b * K + k];
      sp += pred[0] != ep[b];
      d += pred[0] == lab[b];
      if (!ephi.empty() && fold == 1 && b < 8) {
        for (int k = 0; k < K; ++k)
          for (int q = 0; q < D; ++q) sai_phi += g3_dbg_phi[k][q] != ephi[((size_t)b * K + k) * D + q];
        ++so_phi;
      }
    }
    sai_s += ss; sai_p += sp; dung += d; tong += B; ++nfold;
    std::printf("fold %d: sai_score=%ld sai_pred=%ld acc=%.4f\n", fold, ss, sp, double(d) / B);
  }
  std::printf("T=%d fold=%ld trial=%ld sai_score=%ld sai_pred=%ld loi_nap=%ld acc_TB_trial=%.4f phi: %ld luot thu, sai=%ld\n",
              T, nfold, tong, sai_s, sai_p, loi_nap, double(dung) / (tong ? tong : 1), so_phi, sai_phi);
  const bool ok = !sai_s && !sai_p && !loi_nap && nfold > 0 && !sai_phi && (ephi.empty() || so_phi == 8);
  std::printf(ok ? "TB DAT\n" : "TB KHONG DAT\n");
  return ok ? 0 : 1;
}
