import argparse
import csv
import pandas as pd
from typing import List


def parse_keep_cols(s: str) -> List[str]:
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def process_file(input_path: str, output_path: str, text_col: str, label_col: str, keep_cols: List[str], chunksize: int = 100000):
    first_chunk = True
    reader_opts = dict(sep=",", engine="python", quoting=csv.QUOTE_MINIMAL)

    for chunk in pd.read_csv(input_path, chunksize=chunksize, **reader_opts):
        # ensure required columns exist
        if not (text_col in chunk.columns and label_col in chunk.columns):
            raise ValueError(f"Required columns not found in CSV: expected '{text_col}' and '{label_col}'")

        # select columns that exist in this chunk
        cols = [c for c in [text_col, label_col] + keep_cols if c in chunk.columns]

        df = chunk[cols].copy()

        # drop rows missing text or label
        df = df.dropna(subset=[text_col, label_col])

        # attempt to convert label to integer; drop rows that fail
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)

        # rename text and label columns to canonical names
        rename_map = {text_col: "text", label_col: "label"}
        df = df.rename(columns=rename_map)

        # write to CSV (append after first chunk)
        if first_chunk:
            df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
            first_chunk = False
        else:
            df.to_csv(output_path, index=False, header=False, mode="a", quoting=csv.QUOTE_MINIMAL)


def main():
    parser = argparse.ArgumentParser(description="Preprocess sarcasm CSV to keep only text+label and optional metadata")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to cleaned output CSV")
    parser.add_argument("--text-col", default="comment", help="Name of the text column (default: 'comment')")
    parser.add_argument("--label-col", default="label", help="Name of the label column (default: 'label')")
    parser.add_argument("--keep-cols", default="", help="Comma-separated list of additional columns to keep (optional)")
    parser.add_argument("--chunksize", type=int, default=100000, help="Pandas read_csv chunksize (default: 100000)")

    args = parser.parse_args()

    keep_cols = parse_keep_cols(args.keep_cols)

    process_file(args.input, args.output, args.text_col, args.label_col, keep_cols, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
