"""WP2 - tong hop vong tim kien truc va chon ung vien theo  (chi thu vien chuan).

- Moi ho: chi lay N trial COMPLETE dau tien theo so trial (N = 32 cho HA, 40 cho HB/HC/HD;  diem 5).
- Mat Pareto (max accuracy search_val, min MACs) cho tung ho va gop.
- Ung vien lai = accuracy cao nhat trong HB u HC; ung vien thuan hoc sau = cao nhat trong HD.
  Hoa: moi trial cach accuracy cao nhat < 0,5 diem % coi nhu bang nhau -> chon MACs thap nhat.
  HA chi la tham chieu (khong duoc chon).
Dau ra: <dir>/pareto.md, <dir>/selection.json. Tu choi ghi de (OVERWRITE=1).
Dung: /opt/anaconda3/bin/python3 analyze_wp2_search.py --dir results/wp2_search_2026_10_08
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

N_USE = {"HA": 32, "HB": 40, "HC": 40, "HD": 40}
TIE = 0.5
FBCCA_04 = 12.54          # FBCCA Benchmark 0,4 s tren cung cac fold
MAX_PARAMS, MAX_MACS = 50_000, 5_000_000


def load(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "trials_*_w*.csv"))):
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    out, extra, counts = {}, {}, {}
    for fam in N_USE:
        comp = sorted((r for r in rows if r["family"] == fam and r["status"] == "complete"),
                      key=lambda r: int(r["trial"]))
        nums = [int(r["trial"]) for r in comp]
        if len(nums) != len(set(nums)):
            sys.exit(f"LOI: trung so trial trong {fam}")
        if len(comp) < N_USE[fam]:
            sys.exit(f"LOI: {fam} chi co {len(comp)} trial complete (can {N_USE[fam]})")
        for r in comp:
            r["acc"] = float(r["acc_mean"])
            r["macs_i"] = int(r["macs"])
            r["params_i"] = int(r["n_params"])
            r["folds"] = [float(x) for x in r["acc_folds"].split(";")]
            assert len(r["folds"]) == 5, r
            assert r["params_i"] <= MAX_PARAMS and r["macs_i"] <= MAX_MACS, r
        out[fam], extra[fam] = comp[:N_USE[fam]], comp[N_USE[fam]:]
        counts[fam] = {s: sum(1 for r in rows if r["family"] == fam and r["status"] == s)
                       for s in ("complete", "infeasible", "duplicate")}
    return out, extra, counts


def front(rs):
    return [r for r in rs if not any(q is not r and q["acc"] >= r["acc"] and q["macs_i"] <= r["macs_i"]
                                     and (q["acc"] > r["acc"] or q["macs_i"] < r["macs_i"]) for q in rs)]


def pick(rs):
    best = max(r["acc"] for r in rs)
    tied = [r for r in rs if best - r["acc"] < TIE]
    return min(tied, key=lambda r: (r["macs_i"], -r["acc"], int(r["trial"]))), best, tied


def row(r):
    return (f"| {r['family']} | {r['trial']} | {r['acc']:.2f} | {statistics.stdev(r['folds']):.2f} | "
            f"{r['params_i']} | {r['macs_i'] / 1e6:.3f} | `{r['params_json']}` |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    md, js = os.path.join(a.dir, "pareto.md"), os.path.join(a.dir, "selection.json")
    if os.path.exists(js) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{js} da ton tai; OVERWRITE=1 de ghi de")
    use, extra, counts = load(a.dir)
    hyb, hb_best, hb_tied = pick(use["HB"] + use["HC"])
    pure, hd_best, hd_tied = pick(use["HD"])
    ha_top = max(use["HA"], key=lambda r: r["acc"])
    allr = use["HB"] + use["HC"] + use["HD"]
    H = "| Họ | Trial | Acc TB 5 fold | SD giữa fold | Tham số | MACs (M) | Cấu hình |\n|---|---|---|---|---|---|---|"
    L = ["# WP2 — Kết quả vòng tìm kiến trúc (Benchmark, 5-fold theo người, 0,4 s, seed 0)", "",
         "Script: `scripts/analyze_wp2_search.py`. Accuracy = TB trên",
         f"`search_val` của 5 fold (35 người, mỗi người một lần), điểm %. Tham chiếu FBCCA cùng fold: {FBCCA_04}%.",
         "Đây là số của vòng tìm kiếm (1 seed, dùng để CHỌN), không phải kết quả LOSO.", "",
         "## Số trial", "", "| Họ | complete | dùng | dư (không dùng) | infeasible | duplicate |", "|---|---|---|---|---|---|"]
    for f in N_USE:
        L.append(f"| {f} | {counts[f]['complete']} | {len(use[f])} | {len(extra[f])} | "
                 f"{counts[f]['infeasible']} | {counts[f]['duplicate']} |")
    L += ["", "## Ứng viên được chọn", "",
          f"- **Lai (H-B ∪ H-C)**: trial {hyb['family']} #{hyb['trial']}, acc {hyb['acc']:.2f}% "
          f"(cao nhất {hb_best:.2f}%; {len(hb_tied)} trial trong khoảng hoà {TIE} điểm → chọn MACs thấp nhất).",
          f"- **Thuần học sâu (H-D)**: trial #{pure['trial']}, acc {pure['acc']:.2f}% "
          f"(cao nhất {hd_best:.2f}%; {len(hd_tied)} trial trong khoảng hoà).",
          f"- Tham chiếu H-A (không chọn): cao nhất trial #{ha_top['trial']}, acc {ha_top['acc']:.2f}%.", "",
          H, row(hyb), row(pure), row(ha_top), ""]
    for f in N_USE:
        fr = sorted(front(use[f]), key=lambda r: r["macs_i"])
        L += [f"## Mặt Pareto {f} ({len(fr)} điểm)", "", H] + [row(r) for r in fr] + [""]
    fr = sorted(front(allr), key=lambda r: r["macs_i"])
    L += [f"## Mặt Pareto gộp H-B, H-C, H-D ({len(fr)} điểm)", "", H] + [row(r) for r in fr] + [""]
    for f in N_USE:
        L += [f"## Top 5 accuracy {f}", "", H] + [row(r) for r in sorted(use[f], key=lambda r: -r["acc"])[:5]] + [""]
    if any(extra.values()):
        L += ["## Trial dư (không dùng để chọn)", "", H] + [row(r) for f in N_USE for r in extra[f]] + [""]
    os.makedirs(a.dir, exist_ok=True)
    open(md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    sel = {k: {"family": r["family"], "trial": int(r["trial"]), "acc_search": r["acc"], "folds": r["folds"],
               "n_params": r["params_i"], "macs": r["macs_i"], "cfg": json.loads(r["params_json"])}
           for k, r in (("hybrid", hyb), ("pure_dl", pure), ("ref_HA", ha_top))}
    sel["rule"] = "; 40 trial complete dau tien moi ho (HA 32); hoa < 0,5 diem -> MACs thap nhat"
    json.dump(sel, open(js, "w"), indent=1, ensure_ascii=False)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
