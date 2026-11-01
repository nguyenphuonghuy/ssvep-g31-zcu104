// Testbench C-sim / cosim: doc input.bin, chay g3_decode tung trial, so score/pred (va phi khi C-sim) voi expected_*.
// Dung: tb <thu_muc_vector> [B=8] [nophi]  (nophi: cosim — RTL khong ghi mang debug phi)   -> in "TB DAT" va tra 0 khi khop het.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "g3_decode.h"
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
  const int B = argc > 2 ? std::atoi(argv[2]) : 8;
  const bool chk_phi = !(argc > 3 && std::string(argv[3]) == "nophi");
  const int TT = S0 + T;
  std::vector<int16_t> in((size_t)B * C * TT);
  std::vector<int64_t> es((size_t)B * K), ep(B), ephi((size_t)B * K * D);
  if (!rd(dir + "/input.bin", in) || !rd(dir + "/expected_score.bin", es) || !rd(dir + "/expected_pred.bin", ep) ||
      !rd(dir + "/expected_phi.bin", ephi)) { std::printf("TB LOI doc file trong %s\n", dir.c_str()); return 2; }
  long sai_s = 0, sai_p = 0, sai_phi = 0;
  for (int b = 0; b < B; ++b) {
    int64_t score[K];
    int32_t pred[1];
    g3_decode(&in[(size_t)b * C * TT], score, pred);
    for (int k = 0; k < K; ++k) sai_s += score[k] != es[(size_t)b * K + k];
    sai_p += pred[0] != ep[b];
    if (chk_phi)
    for (int k = 0; k < K; ++k)
      for (int d = 0; d < D; ++d) sai_phi += g3_dbg_phi[k][d] != ephi[((size_t)b * K + k) * D + d];
    std::printf("trial %d pred=%d (expected %lld)\n", b, pred[0], (long long)ep[b]);
  }
  std::printf(chk_phi ? "" : "(khong so phi)\n");
  std::printf("T=%d B=%d sai_score=%ld sai_pred=%ld sai_phi=%ld\n", T, B, sai_s, sai_p, sai_phi);
  std::printf(sai_s || sai_p || sai_phi ? "TB KHONG DAT\n" : "TB DAT\n");
  return (sai_s || sai_p || sai_phi) ? 1 : 0;
}
