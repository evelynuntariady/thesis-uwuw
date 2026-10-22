"""
Reproducible label processing: majority vote, agregasi polarisasi, dan balancing 1:3 (seperti paper)

Usage:
    python balancing.py --input thesis_full.csv --output balanced.csv
    python balancing.py --input thesis_full.csv --output balanced.csv --ratio 3 --preprocess
"""

import argparse
import ast

import numpy as np
import pandas as pd

LABEL_COLS = [
    "is_noise_or_spam_text",
    "toxicity",
    "polarized",
    "profanity_obscenity",
    "threat_incitement_to_violence",
    "insults",
    "identity_attack",
    "sexually_explicit",
]
POL_COL = "polarized"
TARGET_LABEL = "toxicity"
NOISE_COL = "is_noise_or_spam_text"
RANDOM_STATE = 42


def parse_label_lists(df, cols=LABEL_COLS):
    # "['0', '1']" (string dari CSV) -> ['0', '1']
    df = df.copy()
    for col in cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def validate_label_lists(df, cols=LABEL_COLS):
    # Pastikan semua elemen label bisa di-cast ke int; tampilkan baris bermasalah kalau tidak
    for col in cols:
        for idx, x in df[col].items():
            try:
                [int(i) for i in x]
            except (TypeError, ValueError):
                raise ValueError(f"Label tidak valid di kolom '{col}', baris {idx}: {x!r}")


def majority_vote(x):
    # Ekuivalen dengan round(mean) di notebook: round() Python membulatkan 0.5 ke 0,
    # jadi kalau seri (mis. [0, 1]) hasilnya 0.
    x = [int(i) for i in x]
    return int(np.mean(x) > 0.5)


def apply_majority_vote(df, cols=None, exclude=(POL_COL,)):
    # Default: semua label kecuali polarized (polarized diagregasi terpisah)
    cols = [c for c in (cols or LABEL_COLS) if c not in exclude]
    df = df.copy()
    for col in cols:
        df[col] = df[col].apply(majority_vote)
    return df


def add_polarization_agg(df, col=POL_COL, list_col="polarization_list", avg_col="pol_avg"):
    # Agregasi polarisasi seperti paper: rata-rata label anotator, mis. [1, 0, 0] -> 0.333
    df = df.copy()
    df[list_col] = df[col].apply(lambda xs: [int(v) for v in xs])
    df[avg_col] = df[list_col].apply(lambda xs: float(np.mean(xs)))
    return df


def filter_noise(df, noise_col=NOISE_COL):
    # Buang teks noise/spam (hasil majority vote) dan baris kosong
    return df[df[noise_col] == 0].dropna()


def balance_undersample(df, label_col=TARGET_LABEL, ratio=3, random_state=RANDOM_STATE):
    # Undersampling kelas negatif sampai neg:pos = ratio:1 (paper: 1:3)
    pos = df[df[label_col] == 1]
    neg_all = df[df[label_col] == 0]
    n_neg = ratio * len(pos)
    if n_neg > len(neg_all):
        raise ValueError(f"Kelas negatif cuma {len(neg_all)} baris, butuh {n_neg} untuk rasio 1:{ratio}")
    neg = neg_all.sample(n=n_neg, random_state=random_state)
    return pd.concat([pos, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_dataset(df, ratio=3, random_state=RANDOM_STATE, preprocess=False):
    # sesuai yg sekarang di notebook: parse label, validasi label, majority vote, filter noise, agregasi polarisasi, dan balancing
    df = parse_label_lists(df)
    validate_label_lists(df)
    df = apply_majority_vote(df)
    df = filter_noise(df)
    df = add_polarization_agg(df)

    if preprocess:
        from preprocess import SOURCE_COL, TARGET_COL, preprocess_text
        df[TARGET_COL] = df[SOURCE_COL].fillna("").astype(str).apply(preprocess_text)

    return balance_undersample(df, ratio=ratio, random_state=random_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path ke thesis_full.csv")
    parser.add_argument("--output", default="balanced.csv")
    parser.add_argument("--ratio", type=int, default=3, help="rasio neg:pos (default 3 -> 1:3)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--preprocess", action="store_true", help="jalankan juga preprocess_text dari preprocess.py")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = prepare_dataset(df, ratio=args.ratio, random_state=args.seed, preprocess=args.preprocess)
    df.to_csv(args.output, index=False)

    print(f"Balanced {len(df)} rows -> {args.output}")
    print(df[TARGET_LABEL].value_counts().to_string())


if __name__ == "__main__":
    main()
