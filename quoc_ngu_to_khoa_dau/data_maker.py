from quoc_ngu_to_khoa_dau.rule_based import encode_custom
import os
import csv

BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "data", "all-vietnamese-syllables.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "all-vietnamese-syllables-encoded.csv")


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(
        OUTPUT_PATH, "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        for raw in fin:
            word = raw.strip()
            if not word:
                continue
            writer.writerow([word, encode_custom(word)])


if __name__ == "__main__":
    main()
