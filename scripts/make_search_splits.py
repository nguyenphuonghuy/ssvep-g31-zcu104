"""Tao file chia 5-fold theo nguoi tren Benchmark cho vong tim kien truc WP2 (chi thu vien chuan).


- 35 nguoi hoan vi bang random.Random(SEED), chia lien tiep thanh 5 fold x 7 nguoi.
- Moi fold: 7 nguoi "danh gia tim kiem" (search-val), 28 nguoi con lai; trong 28 nguoi tach tat dinh
  4 nguoi "dung som" (early-stop) bang random.Random(SEED + fold), 24 nguoi train.
- Moi nguoi la search-val dung mot lan. File giong nhau cho moi ho, moi trial, moi seed.
Tu choi ghi de file da co (OVERWRITE=1 de ghi de).

Dung: /opt/anaconda3/bin/python3 make_search_splits.py
"""
import hashlib
import json
import os
import random
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "configs", "splits_benchmark_search.json")
SEED = 20261008
N_SUBJ, N_FOLD, N_STOP = 35, 5, 4


def main():
    if os.path.exists(OUT) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{OUT} da ton tai; OVERWRITE=1 de ghi de")
    subs = list(range(1, N_SUBJ + 1))
    perm = subs[:]
    random.Random(SEED).shuffle(perm)
    k = N_SUBJ // N_FOLD
    folds = []
    for f in range(N_FOLD):
        val = sorted(perm[f * k:(f + 1) * k])
        rest = [s for s in subs if s not in val]
        stop = sorted(random.Random(SEED + f + 1).sample(rest, N_STOP))
        train = [s for s in rest if s not in stop]
        folds.append({"fold": f, "search_val": val, "early_stop": stop, "train": train})
    # tu kiem
    allval = sorted(s for fo in folds for s in fo["search_val"])
    assert allval == subs, "moi nguoi phai la search-val dung mot lan"
    for fo in folds:
        parts = fo["search_val"] + fo["early_stop"] + fo["train"]
        assert sorted(parts) == subs and len(set(parts)) == N_SUBJ, fo["fold"]
        assert len(fo["train"]) == N_SUBJ - k - N_STOP
    body = {"dataset": "benchmark", "n_subjects": N_SUBJ, "seed": SEED,
            "method": "random.Random(seed).shuffle(1..35) -> 5 khoi lien tiep x 7 nguoi = search_val; "
                      "early_stop = random.Random(seed+fold+1).sample(phan con lai, 4); train = phan con lai",
            "created": "2026-10-08", "purpose": "WP2 tim kien truc",
            "folds": folds}
    digest = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()
    body["sha256_folds"] = digest
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(body, f, indent=1)
    for fo in folds:
        print(f"fold {fo['fold']}: search_val={fo['search_val']} early_stop={fo['early_stop']} "
              f"train={len(fo['train'])} nguoi")
    print("sha256_folds", digest)
    print("ghi", OUT)


if __name__ == "__main__":
    main()
