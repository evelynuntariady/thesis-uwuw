"""
Bandingkan model teks-only vs model metadata pada testcase yang berpotensi
menimbulkan perbedaan label antar annotator.

Setiap teks di testcase dipasangkan dengan SEMUA profil annotator asli dari
dataset explode. Model metadata memprediksi per annotator, model teks-only
hanya sekali per teks (prediksinya sama untuk semua annotator). Dari situ
dilihat apakah model metadata bisa "mendengar" pola minoritas (annotator
in-group, lihat kolom minority_filter) yang tertutup oleh pola mayoritas.

Kolom focus_label di testcase menentukan label yang dianalisis:
  - toxicity                         -> Stage 1 (butuh --meta-s1 dan --text-s1)
  - insults, identity_attack,
    threat_incitement_to_violence,
    profanity_obscenity,
    sexually_explicit                -> Stage 2 (butuh --meta-s2 dan --text-s2)
Kalau kolom focus_label tidak ada, semua case dianggap "toxicity".

Dua format testcase didukung (dideteksi otomatis):
  1. Cross join (testcases_disagreement.csv, testcases_toxicity_type.csv)
     satu baris per teks + kolom minority_filter/expected_majority/expected_minority;
     tiap teks dipasangkan dengan semua annotator.
  2. Pasangan (testcases_text_metadata.csv)
     satu baris per (teks, annotator) dengan kolom annotator_id + metadata + label
     ekspektasi (kolom bernama sama dengan focus_label, default "toxicity").
     Kolom polarized opsional; kalau tidak ada dipakai --polarized.

Input model disamakan dengan notebook training:
  - teks-only S1 : "Nilai rata-rata polarisasi: {pol_avg:.2f}" + processed_text, max_length=128
  - metadata  S1 : "Polarisasi: .. Etnis: .. ... Preferensi politik: .." + processed_text, max_length=256
  - teks-only S2 : processed_text, max_length=128
  - metadata  S2 : metadata tanpa Polarisasi + processed_text, max_length=256

Model bisa berupa repo id Hugging Face, folder, atau .zip. Default-nya diambil dari
Hugging Face: evelnap/indobertweet-text-only-s1/-s2 dan evelnap/indobertweet-metadata-s1/-s2.

Usage (Colab / lokal):
    python compare_models.py --cases testcases_text_metadata.csv --out results_pairs
    python compare_models.py --cases testcases_toxicity_type.csv --out results_type

    python compare_models.py --cases testcases_toxicity_type.csv --dry-run   # cek format input tanpa load model
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1] / "reusable-scripts"))
from preprocess import preprocess_text  # noqa: E402

META_COLS = [
    "ethnicity", "religion", "disability", "lgbt", "gender", "age", "domisili",
    "pendidikan terakhir", "status pekerjaan", "president vote leaning",
]
LABEL_COLS_S2 = [
    "insults", "identity_attack", "threat_incitement_to_violence",
    "profanity_obscenity", "sexually_explicit",
]
ALL_LABELS = ["toxicity"] + LABEL_COLS_S2

# Model IndoBERTweet di Hugging Face (default --text-s1 / --text-s2 / --meta-s1 / --meta-s2)
TEXT_S1_REPO = "evelnap/indobertweet-text-only-s1"
TEXT_S2_REPO = "evelnap/indobertweet-text-only-s2"
META_S1_REPO = "evelnap/indobertweet-metadata-s1"
META_S2_REPO = "evelnap/indobertweet-metadata-s2"


# ---------------------------------------------------------------- input format
def text_only_prefix(pol_avg):
    return f"Nilai rata-rata polarisasi: {pol_avg:.2f}"


def metadata_string(row, with_polarized=True):
    meta = f"Polarisasi: {row['polarized']}. " if with_polarized else ""
    return meta + (
        f"Etnis: {row['ethnicity']}. "
        f"Agama: {row['religion']}. "
        f"Disabilitas: {row['disability']}. "
        f"LGBT: {row['lgbt']}. "
        f"Gender: {row['gender']}. "
        f"Usia: {row['age']}. "
        f"Domisili: {row['domisili']}. "
        f"Pendidikan: {row['pendidikan terakhir']}. "
        f"Pekerjaan: {row['status pekerjaan']}. "
        f"Preferensi politik: {row['president vote leaning']}."
    )


# ---------------------------------------------------------------- data
def load_cases(path):
    cases = pd.read_csv(path)
    if "focus_label" not in cases.columns:
        cases["focus_label"] = "toxicity"
    unknown = set(cases["focus_label"]) - set(ALL_LABELS)
    if unknown:
        raise ValueError(f"focus_label tidak dikenal: {unknown}")
    cases["processed_text"] = cases["text"].apply(preprocess_text)
    return cases


def is_paired(cases):
    return "annotator_id" in cases.columns and set(META_COLS) <= set(cases.columns)


def build_paired(rows, profiles, default_polarized):
    """Format pasangan: baris sudah berisi annotator + metadata + label ekspektasi."""
    pairs = rows.copy()
    if "polarized" not in pairs.columns:
        pairs["polarized"] = default_polarized
    missing = set(pairs["focus_label"]) - set(pairs.columns)
    if missing:
        raise ValueError(f"kolom label ekspektasi tidak ada: {missing}")
    pairs["expected"] = [int(pairs.at[i, l]) for i, l in pairs["focus_label"].items()]

    # Pakai string metadata persis seperti di data training (mis. 'Banda Aceh ' masih ada spasi,
    # gender sudah di-strip), supaya input model metadata sama dengan saat training.
    known = pairs["annotator_id"].isin(profiles["annotator_id"])
    prof = profiles.set_index("annotator_id")
    for col in META_COLS:
        train_vals = pairs.loc[known, "annotator_id"].map(prof[col])
        diff = (pairs.loc[known, col].astype(str) != train_vals.astype(str))
        if diff.any():
            print(f"[INFO] {col}: {diff.sum()} nilai diganti mengikuti data training "
                  f"(contoh {pairs.loc[known, col][diff].iloc[0]!r} -> {train_vals[diff].iloc[0]!r})")
        pairs.loc[known, col] = train_vals
    if (~known).any():
        print(f"[INFO] annotator {sorted(pairs.loc[~known, 'annotator_id'].unique())} tidak ada di data training, "
              "metadata dipakai apa adanya dari file testcase")

    cases = pairs.drop_duplicates("case_id")[["case_id", "text", "focus_label", "polarized", "processed_text"]]
    return cases.reset_index(drop=True), pairs.reset_index(drop=True)


def load_profiles(path):
    """Profil unik per annotator, difilter sama seperti notebook metadata."""
    df = pd.read_csv(path)
    df = df[(df["is_noise_or_spam_text"] == 0) & (df["is_valid"] == 1)].dropna()
    profiles = df[["annotator_id"] + META_COLS].drop_duplicates("annotator_id")
    return profiles.sort_values("annotator_id").reset_index(drop=True)


def build_pairs(cases, profiles):
    """Cross join case x annotator + tandai annotator in-group per case."""
    pairs = cases.merge(profiles, how="cross")
    pairs["in_group"] = False
    for case_id, flt in cases.set_index("case_id")["minority_filter"].items():
        ids = set(profiles.query(flt)["annotator_id"])
        if not ids:
            print(f"[WARN] {case_id}: filter '{flt}' tidak cocok dengan annotator mana pun")
        mask = (pairs["case_id"] == case_id) & pairs["annotator_id"].isin(ids)
        pairs.loc[mask, "in_group"] = True
    pairs["expected"] = np.where(pairs["in_group"], pairs["expected_minority"], pairs["expected_majority"]).astype(int)
    return pairs


# ---------------------------------------------------------------- model
def resolve_model_dir(path):
    """Terima repo id Hugging Face (mis. evelnap/indobertweet-text-only-s1),
    folder model, atau .zip hasil shutil.make_archive dari notebook."""
    if not Path(path).exists():
        return path   # dianggap repo id Hugging Face, di-download oleh from_pretrained
    path = Path(path)
    if path.suffix == ".zip":
        target = path.with_name(path.stem + "_extracted")
        if not target.exists():
            print(f"Extract {path} -> {target}")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
        path = target
    # model final ada di folder paling dangkal yang punya config.json (bukan checkpoint-*)
    configs = sorted(path.rglob("config.json"), key=lambda p: len(p.parts))
    if not configs:
        raise FileNotFoundError(f"config.json tidak ditemukan di {path}")
    return configs[0].parent


class Predictor:
    def __init__(self, model_path, device, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = resolve_model_dir(model_path)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.multi_label = self.model.config.problem_type == "multi_label_classification"
        print(f"Loaded {model_dir} ({'multi-label' if self.multi_label else 'single-label'})")

    def predict(self, first, second=None, max_length=128):
        """Return prob kelas toxic (S1) atau prob per label (S2)."""
        out = []
        for i in range(0, len(first), self.batch_size):
            a = first[i:i + self.batch_size]
            b = second[i:i + self.batch_size] if second is not None else None
            enc = self.tokenizer(
                a, b,
                truncation="only_second" if b is not None else True,
                max_length=max_length, padding="max_length", return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            probs = self.torch.sigmoid(logits) if self.multi_label else self.torch.softmax(logits, dim=-1)[:, 1]
            out.append(probs.cpu().numpy())
        return np.concatenate(out)


def run_stage1(cases, pairs, meta_path, text_path, device, batch_size):
    text_m = Predictor(text_path, device, batch_size)
    prefix = [text_only_prefix(float(v)) for v in cases["polarized"]]
    cases["text_prob_toxicity"] = text_m.predict(prefix, cases["processed_text"].tolist(), max_length=128)
    del text_m

    meta_m = Predictor(meta_path, device, batch_size)
    meta = pairs.apply(metadata_string, axis=1).tolist()
    pairs["meta_prob_toxicity"] = meta_m.predict(meta, pairs["processed_text"].tolist(), max_length=256)
    del meta_m


def run_stage2(cases, pairs, meta_path, text_path, device, batch_size):
    text_m = Predictor(text_path, device, batch_size)
    probs = text_m.predict(cases["processed_text"].tolist(), max_length=128)
    for i, c in enumerate(LABEL_COLS_S2):
        cases[f"text_prob_{c}"] = probs[:, i]
    del text_m

    meta_m = Predictor(meta_path, device, batch_size)
    meta = pairs.apply(metadata_string, axis=1, with_polarized=False).tolist()
    probs = meta_m.predict(meta, pairs["processed_text"].tolist(), max_length=256)
    for i, c in enumerate(LABEL_COLS_S2):
        pairs[f"meta_prob_{c}"] = probs[:, i]
    del meta_m


# ---------------------------------------------------------------- analysis
def add_focus_columns(pairs, threshold):
    """Ambil prob untuk focus_label tiap case -> kolom text_prob/meta_prob/text_pred/meta_pred."""
    pairs["text_prob"] = [pairs.at[i, f"text_prob_{l}"] for i, l in pairs["focus_label"].items()]
    pairs["meta_prob"] = [pairs.at[i, f"meta_prob_{l}"] for i, l in pairs["focus_label"].items()]
    pairs["text_pred"] = (pairs["text_prob"] >= threshold).astype(int)
    pairs["meta_pred"] = (pairs["meta_prob"] >= threshold).astype(int)
    return pairs


def summarize_paired(pairs):
    """Per case: annotator yang diharapkan tersinggung (expected=1) vs yang tidak (expected=0)."""
    rows = []
    for case_id, g in pairs.groupby("case_id", sort=False):
        pos, neg = g[g["expected"] == 1], g[g["expected"] == 0]
        first = g.iloc[0]
        rows.append({
            "case_id": case_id,
            "focus_label": first["focus_label"],
            "annotator_tersinggung": ",".join(pos["annotator_id"].astype(str)),
            "annotator_tidak": ",".join(neg["annotator_id"].astype(str)),
            "text_prob": round(first["text_prob"], 4),
            "text_pred": int(first["text_pred"]),
            "meta_prob_tersinggung": round(pos["meta_prob"].mean(), 4),
            "meta_prob_tidak": round(neg["meta_prob"].mean(), 4),
            "meta_prob_gap": round(pos["meta_prob"].mean() - neg["meta_prob"].mean(), 4),
            "meta_pred_tersinggung": ",".join(pos["meta_pred"].astype(str)),
            "meta_pred_tidak": ",".join(neg["meta_pred"].astype(str)),
            "text_benar": f"{int((g['text_pred'] == g['expected']).sum())}/{len(g)}",
            "meta_benar": f"{int((g['meta_pred'] == g['expected']).sum())}/{len(g)}",
            # metadata memberi prediksi berbeda antar annotator untuk teks yang sama
            "meta_separates": g["meta_pred"].nunique() > 1,
            # prob annotator yang tersinggung lebih tinggi dari yang tidak
            "gap_direction_ok": pos["meta_prob"].mean() > neg["meta_prob"].mean(),
            # semua annotator di case ini diprediksi benar
            "text_all_correct": bool((g["text_pred"] == g["expected"]).all()),
            "meta_all_correct": bool((g["meta_pred"] == g["expected"]).all()),
        })
    return pd.DataFrame(rows)


def paired_scores(pairs):
    subsets = {
        "semua annotator": pairs,
        "annotator tersinggung (expected=1)": pairs[pairs["expected"] == 1],
        "annotator tidak tersinggung (expected=0)": pairs[pairs["expected"] == 0],
    }
    scores = pd.DataFrame({k: accuracy_table(v) for k, v in subsets.items()}).T
    scores.index.name = "subset"
    scores["n_pairs"] = scores["n_pairs"].astype(int)
    return scores.reset_index()


def print_paired_report(summary, scores):
    n = len(summary)
    print("\n" + "=" * 90)
    print("RINGKASAN PER CASE")
    print("=" * 90)
    cols = ["case_id", "annotator_tersinggung", "annotator_tidak", "text_pred",
            "meta_pred_tersinggung", "meta_pred_tidak", "meta_prob_gap", "text_benar", "meta_benar"]
    print(summary[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("AGREGAT")
    print("=" * 90)
    print(f"Metadata membedakan prediksi antar annotator : {summary['meta_separates'].sum()}/{n}")
    print(f"Prob annotator tersinggung > tidak tersinggung: {summary['gap_direction_ok'].sum()}/{n}")
    print(f"Semua annotator benar - teks-only            : {summary['text_all_correct'].sum()}/{n}")
    print(f"Semua annotator benar - metadata             : {summary['meta_all_correct'].sum()}/{n}")
    print(f"Rata-rata gap prob (tersinggung - tidak)     : {summary['meta_prob_gap'].mean():.4f}")
    print("\nAkurasi terhadap label ekspektasi per annotator:")
    print(scores.to_string(index=False))


def summarize_cases(pairs):
    rows = []
    for case_id, g in pairs.groupby("case_id", sort=False):
        ig, og = g[g["in_group"]], g[~g["in_group"]]
        first = g.iloc[0]
        exp_maj, exp_min = int(first["expected_majority"]), int(first["expected_minority"])
        text_pred = int(first["text_pred"])
        ig_rate = ig["meta_pred"].mean() if len(ig) else np.nan
        og_rate = og["meta_pred"].mean()
        ig_label = int(ig_rate >= 0.5) if len(ig) else np.nan
        og_label = int(og_rate >= 0.5)
        gap = ig["meta_prob"].mean() - og["meta_prob"].mean() if len(ig) else np.nan
        rows.append({
            "case_id": case_id,
            "focus_label": first["focus_label"],
            "target_group": first["target_group"],
            "n_in_group": len(ig),
            "expected_majority": exp_maj,
            "expected_minority": exp_min,
            "text_prob": round(first["text_prob"], 4),
            "text_pred": text_pred,
            "meta_prob_in_group": round(ig["meta_prob"].mean(), 4),
            "meta_prob_out_group": round(og["meta_prob"].mean(), 4),
            "meta_prob_gap": round(gap, 4),
            "meta_prob_std_all": round(g["meta_prob"].std(), 4),
            "meta_rate_in_group": round(ig_rate, 4),
            "meta_rate_out_group": round(og_rate, 4),
            "meta_label_in_group": ig_label,
            "meta_label_out_group": og_label,
            "n_annotator_meta_differs_from_text": int((g["meta_pred"] != text_pred).sum()),
            # text-only mengikuti pola mayoritas?
            "text_follows_majority": text_pred == exp_maj,
            # metadata menghasilkan label in-group sesuai ekspektasi minoritas?
            "meta_captures_minority": ig_label == exp_min,
            # arah selisih prob sesuai hipotesis (minoritas lebih tinggi / lebih rendah)?
            "gap_direction_ok": bool(np.sign(gap) == np.sign(exp_min - exp_maj)) if len(ig) else False,
            # metadata memberi label berbeda untuk in-group vs out-group
            "meta_separates_groups": ig_label != og_label,
        })
    return pd.DataFrame(rows)


def accuracy_table(sub):
    return pd.Series({
        "n_pairs": len(sub),
        "acc_text_only": round((sub["text_pred"] == sub["expected"]).mean(), 4),
        "acc_metadata": round((sub["meta_pred"] == sub["expected"]).mean(), 4),
    })


def annotator_level_scores(pairs):
    """Akurasi vs label ekspektasi per annotator: in-group -> expected_minority, out-group -> expected_majority."""
    subsets = {"semua annotator": pairs, "in-group (minoritas)": pairs[pairs["in_group"]], "out-group": pairs[~pairs["in_group"]]}
    overall = pd.DataFrame({k: accuracy_table(v) for k, v in subsets.items()}).T
    overall.index.name = "subset"
    overall["n_pairs"] = overall["n_pairs"].astype(int)

    per_label = pairs.groupby(["focus_label", "in_group"]).apply(accuracy_table).reset_index()
    per_label["n_pairs"] = per_label["n_pairs"].astype(int)
    per_label["in_group"] = per_label["in_group"].map({True: "in-group", False: "out-group"})
    return overall.reset_index(), per_label


def print_report(summary, overall, per_label):
    n = len(summary)
    print("\n" + "=" * 90)
    print("RINGKASAN PER CASE")
    print("=" * 90)
    cols = ["case_id", "focus_label", "target_group", "n_in_group", "expected_majority", "expected_minority",
            "text_pred", "meta_label_in_group", "meta_label_out_group", "meta_prob_gap"]
    print(summary[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("AGREGAT")
    print("=" * 90)
    print(f"Teks-only mengikuti label mayoritas      : {summary['text_follows_majority'].sum()}/{n}")
    print(f"Metadata menangkap label minoritas       : {summary['meta_captures_minority'].sum()}/{n}")
    print(f"Metadata memisahkan in-group vs out-group: {summary['meta_separates_groups'].sum()}/{n}")
    print(f"Arah gap prob sesuai hipotesis           : {summary['gap_direction_ok'].sum()}/{n}")
    print(f"Rata-rata |gap| prob in vs out           : {summary['meta_prob_gap'].abs().mean():.4f}")

    print("\nPer focus_label:")
    by_label = summary.groupby("focus_label").agg(
        n_case=("case_id", "size"),
        text_follows_majority=("text_follows_majority", "sum"),
        meta_captures_minority=("meta_captures_minority", "sum"),
        gap_direction_ok=("gap_direction_ok", "sum"),
        mean_abs_gap=("meta_prob_gap", lambda s: round(s.abs().mean(), 4)),
    )
    print(by_label.to_string())

    print("\nAkurasi terhadap label ekspektasi per annotator:")
    print(overall.to_string(index=False))
    print("\nAkurasi per focus_label x kelompok:")
    print(per_label.to_string(index=False))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(HERE / "testcases_toxicity_type.csv"))
    parser.add_argument("--profiles", default=str(HERE.parent / "data" / "explode_dataset_using_latest_cleaned_text.csv"),
                        help="dataset explode, dipakai untuk mengambil profil annotator")
    parser.add_argument("--meta-s1", default=META_S1_REPO, help="repo HF / folder / zip model metadata stage 1")
    parser.add_argument("--text-s1", default=TEXT_S1_REPO, help="repo HF / folder / zip model teks-only stage 1")
    parser.add_argument("--meta-s2", default=META_S2_REPO, help="repo HF / folder / zip model metadata stage 2")
    parser.add_argument("--text-s2", default=TEXT_S2_REPO, help="repo HF / folder / zip model teks-only stage 2")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--polarized", type=int, default=0,
                        help="nilai polarized kalau testcase format pasangan tidak punya kolom polarized")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="tampilkan contoh input model, tanpa load model")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    profiles = load_profiles(args.profiles)
    paired = is_paired(cases)
    if paired:
        cases, pairs = build_paired(cases, profiles, args.polarized)
        print(f"Format pasangan: {len(cases)} case, {len(pairs)} pasangan (teks, annotator)")
    else:
        pairs = build_pairs(cases, profiles)
        print(f"{len(cases)} case x {len(profiles)} annotator = {len(pairs)} pasangan")
        print(pairs.groupby("case_id", sort=False)["in_group"].sum().rename("n_in_group").to_frame().T.to_string())

    if args.dry_run:
        row = pairs.iloc[0]
        print("\n[teks-only S1]", text_only_prefix(float(row["polarized"])), "||", row["processed_text"])
        print("[metadata  S1]", metadata_string(row), "||", row["processed_text"])
        print("[teks-only S2]", row["processed_text"])
        print("[metadata  S2]", metadata_string(row, with_polarized=False), "||", row["processed_text"])
        return

    need_s1 = (cases["focus_label"] == "toxicity").any()
    need_s2 = cases["focus_label"].isin(LABEL_COLS_S2).any()
    has_s1 = bool(args.meta_s1 and args.text_s1)
    has_s2 = bool(args.meta_s2 and args.text_s2)
    if need_s1 and not has_s1:
        parser.error("ada case focus_label=toxicity -> --meta-s1 dan --text-s1 wajib diisi")
    if need_s2 and not has_s2:
        parser.error("ada case focus_label stage 2 -> --meta-s2 dan --text-s2 wajib diisi")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hanya stage yang dibutuhkan focus_label yang di-download & dijalankan
    if need_s1:
        run_stage1(cases, pairs, args.meta_s1, args.text_s1, device, args.batch_size)
    if need_s2:
        run_stage2(cases, pairs, args.meta_s2, args.text_s2, device, args.batch_size)

    text_cols = [c for c in cases.columns if c.startswith("text_prob_")]
    pairs = pairs.merge(cases[["case_id"] + text_cols], on="case_id")
    pairs = add_focus_columns(pairs, args.threshold)

    os.makedirs(args.out, exist_ok=True)
    pairs.drop(columns=["processed_text"]).to_csv(f"{args.out}/predictions_per_annotator.csv", index=False)

    if paired:
        summary = summarize_paired(pairs)
        scores = paired_scores(pairs)
        summary.to_csv(f"{args.out}/case_summary.csv", index=False)
        scores.to_csv(f"{args.out}/annotator_level_scores.csv", index=False)
        print_paired_report(summary, scores)
    else:
        summary = summarize_cases(pairs)
        overall, per_label = annotator_level_scores(pairs)
        summary.to_csv(f"{args.out}/case_summary.csv", index=False)
        overall.to_csv(f"{args.out}/annotator_level_scores.csv", index=False)
        per_label.to_csv(f"{args.out}/annotator_level_scores_per_label.csv", index=False)
        print_report(summary, overall, per_label)
    print(f"\nHasil tersimpan di {args.out}/")


if __name__ == "__main__":
    main()
