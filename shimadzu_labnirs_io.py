"""
Robust readers for Shimadzu LABNIRS files (.pat, .ext, .omm, .txt)

▸ Requires pandas, numpy, and chardet  (pip install pandas numpy chardet)
▸ Works on plain Python ≥3.8
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import chardet
import configparser
import re

__all__ = ["read_pat", "read_ext", "read_omm", "read_txt"]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _detect_encoding(file: Path, nbytes: int = 4096) -> str:
    """Return best-guess text encoding (falls back to latin-1)."""
    raw = file.read_bytes()[:nbytes]
    guess = chardet.detect(raw).get("encoding", None)
    return guess or "latin-1"


def _is_numeric_line(line: str) -> bool:
    """Is the line purely whitespace-separated floats / ints?"""
    tokens = line.strip().split()
    if not tokens:
        return False
    try:
        _ = [float(tok) for tok in tokens]
        return True
    except ValueError:
        return False


def _load_text_lines(file: Path) -> list[str]:
    enc = _detect_encoding(file)
    return file.read_text(encoding=enc, errors="replace").splitlines()


# ---------------------------------------------------------------------
# individual readers
# ---------------------------------------------------------------------
def read_pat(file_path):
    """
    Custom reader for Shimadzu .pat files.
    Returns: dict {section_name: {key: value, ...}, ...}
    """
    sections = {}
    current_section = None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                section_name = line[1:-1].strip()
                current_section = section_name
                sections[current_section] = {}
            elif '=' in line and current_section:
                key, val = line.split('=', 1)
                sections[current_section][key.strip()] = val.strip()
            # else: skip unstructured line

    return sections


def read_ext(file: str | Path) -> pd.DataFrame | None:
    """
    .ext  (trial & trigger log).  Some labs save these as binary;
    we try text first and otherwise return None.
    Returns a DataFrame if successful, else None.
    """
    file = Path(file)
    try:
        df = pd.read_csv(
            file,
            sep=r"\s+|,|;",          # whitespace or CSV, auto-detect
            engine="python",
            encoding=_detect_encoding(file),
            on_bad_lines="skip",
        )
        return df
    except UnicodeDecodeError:
        error_msg = (f"Could not decode {file.name} as text. "
                     "It may be a binary file; returning None.")
        print(error_msg)
        return None

def read_omm_2(file_path):
    """
    Parse Shimadzu .OMM files exported as text (no header).
    Assumes space-delimited numeric matrix. Just Tries to read all numeric lines.
    Returns a DataFrame with generic column names.
    This is not working as it should
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = [float(val) for val in line.split()]
                data.append(row)
            except ValueError:
                continue  # Skip non-numeric lines
    df = pd.DataFrame(data)
    df.columns = [f"col{i+1}" for i in range(df.shape[1])]
    return df


def read_omm(file: str | Path) -> pd.DataFrame:
    """
    .omm  (optical measurement matrix)
    Handles both:
      • ASCII export (recommended – ‘Output as text (NIRS-SPM)’)  :contentReference[oaicite:0]{index=0}
      • Raw binary header + ASCII numeric body (common in v11.x)
    """
    file = Path(file)
    lines = _load_text_lines(file)

    numeric_rows: list[list[float]] = []
    header_tokens: list[str] | None = None

    # Pass 1 – detect header & numeric lines
    for line in lines:
        if not line.strip():
            continue

        # a) Try header (contains letters & numbers, but NOT all numeric)
        if header_tokens is None and re.search(r"[A-Za-z]", line):
            header_tokens = re.split(r"[,\s]+", line.strip())
            # sanity check – if header is nonsense, forget it
            if sum(tok.replace(".", "", 1).isdigit() for tok in header_tokens) > 0:
                header_tokens = None
            continue

        # b) Pure numeric line?
        if _is_numeric_line(line):
            numeric_rows.append([float(tok) for tok in line.split()])
        # else ignore

    if not numeric_rows:
        raise ValueError(
            f"No numeric data detected in {file.name}. "
            "Confirm that you exported the file as text."
        )

    # Use widest row length to pad short rows
    width = max(len(r) for r in numeric_rows)
    data = np.asarray(
        [r + [np.nan] * (width - len(r)) for r in numeric_rows], dtype=float
    )

    # Column labels
    if header_tokens and len(header_tokens) == width:
        columns = header_tokens
    else:
        columns = [f"col{i+1}" for i in range(width)]

    return pd.DataFrame(data, columns=columns)



def read_txt_fnirs_data(file_path):
    """
    Parse Shimadzu .txt files containing fNIRS data. This is the standard format
    exported by the LABNIRS software for fNIRS measurements. This is working correctly.
    
    Returns a DataFrame with time and HbO/HbR/HbT columns.
    1. Detect the header line containing "Time(sec)".
    2. Determine number of channels from header.
    3. Read data into DataFrame with appropriate column names.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find the header line with "Time(sec)"
    data_start_idx = None
    for i, line in enumerate(lines):
        if "Time(sec)" in line:
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("Data header not found in file.")

    # Read the header and determine number of channels
    header_line = lines[data_start_idx].strip()
    raw_header_parts = re.split(r'\s+', header_line)
    num_header_fields = len(raw_header_parts)

    # Basic columns before Hb values start
    fixed_columns = ['Time(sec)', 'Task', 'Mark', 'Count']
    num_channels = (num_header_fields - len(fixed_columns)) // 3

    # Build full column names
    columns = fixed_columns.copy()
    for ch in range(1, num_channels + 1):
        columns.extend([
            f'ch{ch:02d}_HbO',
            f'ch{ch:02d}_HbR',
            f'ch{ch:02d}_HbT'
        ])

    # Load the data
    data = pd.read_csv(
        file_path,
        sep=r'\s+',
        engine='python',
        skiprows=data_start_idx + 1,
        names=columns,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    return data






"""# ---------------------------------------------------------------------
# quick test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pprint

    p = argparse.ArgumentParser()
    p.add_argument("file", help="any .pat / .ext / .omm / .txt")
    args = p.parse_args()

    ext = Path(args.file).suffix.lower()
    if ext == ".pat":
        pprint.pp(read_pat(args.file))
    elif ext == ".ext":
        print(read_ext(args.file))
    elif ext == ".omm":
        print(read_omm(args.file).head())
    elif ext == ".txt":
        out = read_txt(args.file)
        print(out if isinstance(out, dict) else out.head())
    else:
        p.error("Unsupported extension")"""
