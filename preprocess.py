"""
Reproducible text preprocessing

Usage:
    python preprocess.py --input thesis_full.csv --output preprocessed.csv
"""

import argparse
import html
import re

import emoji
import pandas as pd

SOURCE_COL = "cleaned_text"
TARGET_COL = "processed_text"
EMOJI_LANG = "id"

# Baris dengan label noise yang rusak di CSV: "(['0', '1'], '0')" (posisi, bukan index label)
BAD_ROW_POS = 5389
BAD_ROW_FIX = "[0, 1, 0]"

# Emoticon teks -> kata. Hanya dicocokkan kalau berdiri sendiri (diapit spasi),
# supaya tidak salah tangkap di tengah kata.
emoticon_dict = {
    ":)": "senyum", ":-)": "senyum", "(:": "senyum", ":]": "senyum", "=)": "senyum",
    ":d": "tertawa", ":-d": "tertawa", "xd": "tertawa", "wkwk": "tertawa",
    ":(": "sedih", ":-(": "sedih", "):": "sedih", ":'(": "menangis",
    ";)": "berkedip", ";-)": "berkedip",
    ":p": "menjulurkan lidah", ":-p": "menjulurkan lidah",
    ":o": "terkejut", ":-o": "terkejut",
    ":/": "ragu", ":-/": "ragu",
    ">:(": "marah", "-_-": "kesal", "<3": "cinta", "</3": "patah hati",
}
_emoticon_re = re.compile(
    r"(?<!\S)(" + "|".join(re.escape(k) for k in sorted(emoticon_dict, key=len, reverse=True)) + r")(?!\S)",
    flags=re.IGNORECASE,
)


def convert_emoticons(text):
    return _emoticon_re.sub(lambda m: f" {emoticon_dict[m.group(1).lower()]} ", text)


def convert_emojis(text, language=EMOJI_LANG):
    # 😂 -> "wajah gembira berurai air mata"
    text = emoji.demojize(text, delimiters=("<<", ">>"), language=language)
    return re.sub(r"<<([^<>]+)>>", lambda m: " " + m.group(1).replace("_", " ") + " ", text)


def preprocess_text(text, lowercase=True):
    if not isinstance(text, str):
        return ""

    # 1. decode HTML entities (&amp; -> &) dan hapus tag HTML
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. remove username (@user)
    text = re.sub(r'@\w+', '', text)

    # 3. remove URL
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 4. emoticon -> kata (setelah URL dibuang, supaya "http://" tidak terbaca ":/")
    text = convert_emoticons(text)

    # 5. emoji -> kata (HARUS sebelum langkah 7, karena langkah 7 menghapus simbol/emoji)
    text = convert_emojis(text)

    # 6. remove numbers
    text = re.sub(r'\d+', '', text)

    # 7. remove unwanted symbols (keep .,!?)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)

    # 9. normalize repeated chars (contoh: anjiiing -> anjiing, !!!!! -> !!)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 10. lowercase (optional)
    if lowercase:
        text = text.lower()

    # 11. remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path ke thesis_full.csv")
    parser.add_argument("--output", default="preprocessed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Disimpan sebagai string supaya tetap bisa di-parse dengan ast.literal_eval seperti baris lain
    df.iat[BAD_ROW_POS, df.columns.get_loc("is_noise_or_spam_text")] = BAD_ROW_FIX
    df[TARGET_COL] = df[SOURCE_COL].fillna("").astype(str).apply(preprocess_text)
    df.to_csv(args.output, index=False)

    print(f"Preprocessed {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
