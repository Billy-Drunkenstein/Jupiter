from pathlib import Path
import os
import json
import numpy as np
import pandas as pd


from SigTest.config import (
    JOIN_KEYS,
    DEFAULT_FEATURE_PATH,
    DEFAULT_TARGET_PATH,
    DEFAULT_META_PATH,
    DEFAULT_REF_HEADERS_PATH,
    DEFAULT_TARGET_COLS,
    DEFAULT_TARGET_MAPPING_PATH,
)
from SigTest.Loaders.raw import FeatherLoader


def load_target_mapping(path: Path = DEFAULT_TARGET_MAPPING_PATH) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def write_headers(path: Path, headers: list[str]) -> None:
    with open(path, "w", encoding = "utf-8") as f:
        for s in headers:
            f.write(s + "\n")


def build_XX(X: np.ndarray, p: int) -> np.ndarray:
    if X.shape[1] != p:
        raise ValueError(f"X columns mismatch: X {X.shape[1]} cols vs expected {p} cols")

    n = X.shape[0]
    XX = np.zeros((p + 1, p + 1), dtype = np.float32)

    XT1 = X.sum(axis = 0) / n
    XTX = X.T @ X

    XX[0, 0] = n
    XX[0, 1:] = XT1
    XX[1:, :p] = XTX

    return XX


def build_XY(X: np.ndarray, y: np.ndarray) -> dict:
    n = X.shape[0]
    sumX = X.sum(axis = 0)
    sumXX = (X * X).sum(axis = 0)
    sumY = y.sum()
    sumYY = (y * y).sum()
    sumXY = X.T @ y

    num = n * sumXY - sumX * sumY
    den = np.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))
    cor = num / den

    return {
        "N": n,
        "X": sumX,
        "XX": sumXX,
        "Y": sumY,
        "YY": sumYY,
        "XY": sumXY,
        "cor": cor,
    }


def get_existing_prefixes(corr_path: Path) -> list[str]:
    if not corr_path.exists():
        return []
    
    df = pd.read_csv(corr_path, nrows = 0)
    prefixes = set()
    for col in df.columns:
        if col.startswith("Y") and "_" in col:
            prefix = col.split("_")[0]
            prefixes.add(prefix)
    return list(prefixes)


def check_exists(
    date: int,
    output_path: Path,
    target_cols: list[str],
    target_mapping: dict,
) -> dict:
    headers_path = output_path / f"FeatureHeaders.{date}"
    xx_path = output_path / f"XX.{date}.csv"
    corr_path = output_path / f"FeatureCorr.{date}.csv"

    files_exist = {
        "headers": headers_path.exists(),
        "xx": xx_path.exists(),
        "corr": corr_path.exists(),
    }
    n_exists = sum(files_exist.values())

    if n_exists == 0:
        needed_prefixes = [target_mapping[t] for t in target_cols]
        return {
            "files": "missing",
            "existing_prefixes": [],
            "missing_prefixes": needed_prefixes,
        }

    if n_exists != 3:
        missing = [k for k, v in files_exist.items() if not v]
        present = [k for k, v in files_exist.items() if v]
        raise RuntimeError(
            f"Partial outputs for date = {date}. Present: {present}, Missing: {missing}"
        )

    existing_prefixes = get_existing_prefixes(corr_path)
    needed_prefixes = [target_mapping[t] for t in target_cols]
    missing_prefixes = [p for p in needed_prefixes if p not in existing_prefixes]

    if not missing_prefixes:
        return {
            "files": "complete",
            "existing_prefixes": existing_prefixes,
            "missing_prefixes": [],
        }

    return {
        "files": "complete",
        "existing_prefixes": existing_prefixes,
        "missing_prefixes": missing_prefixes,
    }


def build_meta_matrices(
    date: int,
    target_cols: list[str] = DEFAULT_TARGET_COLS,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    target_path: Path = DEFAULT_TARGET_PATH,
    output_path: Path = DEFAULT_META_PATH,
    ref_headers_path: Path = DEFAULT_REF_HEADERS_PATH,
    target_mapping_path: Path = DEFAULT_TARGET_MAPPING_PATH,
) -> None:
    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f"output_path must exist and be a directory: {output_path}")

    target_mapping = load_target_mapping(target_mapping_path)

    for t in target_cols:
        if t not in target_mapping:
            raise ValueError(f"Unknown target '{t}' not in target_mapping")

    status = check_exists(date, output_path, target_cols, target_mapping)

    if not status["missing_prefixes"]:
        return

    loader = FeatherLoader(
        feature_path = feature_path,
        target_path = target_path,
        ref_headers_path = ref_headers_path,
    )

    features_df, targets_df = loader.load_day(date, features = "full", targets = None)
    ref_headers = loader.ref_headers
    p = len(ref_headers)

    headers_tmp = output_path / f"FeatureHeaders.{date}.tmp"
    headers_final = output_path / f"FeatureHeaders.{date}"
    xx_tmp = output_path / f"XX.{date}.csv.tmp"
    xx_final = output_path / f"XX.{date}.csv"
    corr_tmp = output_path / f"FeatureCorr.{date}.csv.tmp"
    corr_final = output_path / f"FeatureCorr.{date}.csv"

    if status["files"] == "missing":
        write_headers(headers_tmp, ref_headers)

        X_full = features_df[ref_headers].to_numpy(dtype = np.float32, copy = False)
        XX = build_XX(X_full, p)
        np.savetxt(xx_tmp, XX, delimiter = ",")

        corr_df = pd.DataFrame({
            "ukey": np.full(p, 99999999),
            "fid": [f"F{i}" for i in range(p)],
        })
    else:
        corr_df = pd.read_csv(corr_final)

    missing_targets = [t for t in target_cols if target_mapping[t] in status["missing_prefixes"]]

    for tcol in missing_targets:
        prefix = target_mapping[tcol]
        X, y = loader.merge_target(features_df, targets_df, tcol, feature_cols = "full")

        if X.shape[1] != p:
            raise ValueError(f"Header mismatch: expected {p} cols, found {X.shape[1]}")

        stats = build_XY(X, y)
        corr_df[f"{prefix}_N"] = stats["N"]
        corr_df[f"{prefix}_X"] = stats["X"]
        corr_df[f"{prefix}_Y"] = stats["Y"]
        corr_df[f"{prefix}_XX"] = stats["XX"]
        corr_df[f"{prefix}_XY"] = stats["XY"]
        corr_df[f"{prefix}_YY"] = stats["YY"]
        corr_df[f"{prefix}_cor"] = stats["cor"]

    corr_df.to_csv(corr_tmp, index = False)

    if status["files"] == "missing":
        os.replace(headers_tmp, headers_final)
        os.replace(xx_tmp, xx_final)
    os.replace(corr_tmp, corr_final)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required = True, type = int)
    ap.add_argument("--output_path", default = str(DEFAULT_META_PATH))
    args = ap.parse_args()

    build_meta_matrices(
        date = args.date,
        output_path = Path(args.output_path),
    )


if __name__ == "__main__":
    main()
