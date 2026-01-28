import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import time
import re
from tqdm import tqdm
import json
import pyarrow.feather as feather
from typing import Sequence


base_dir = "/gpfs/hddfs/shared/tzheng_ryin"
features_dir = "cnfut/cnfut_snap_pool_feather"
targets_dir = "cnfut/cnfut_snap_y_feather"
ref_headers_dir = "cnfut/FeatureHeaders.00000000"
output_dir = "cnfut_meta_matrices"

features_path = os.path.join(base_dir, features_dir)
targets_path = os.path.join(base_dir, targets_dir)
reference_headers_path = os.path.join(base_dir, ref_headers_dir)
output_path = Path(os.path.join(base_dir, output_dir))


def read_headers(path, encoding = "utf-8"):
    with open(path, "r", encoding = encoding) as f:
        return [line.strip("\n") for line in f]

def write_headers(path, headers, encoding = "utf-8", newline = "\n"):
    with open(path, "w", encoding = encoding) as f:
        for s in headers:
            f.write(s + newline)


def build_feature_headers(features_path, prefix = "cnfut", memory_map = True):
    tbl = feather.read_table(features_path, memory_map = memory_map)
    return [c for c in tbl.schema.names if c.startswith(prefix)]


# XX = [[    n    , mean(X) ],
#       [ X.T @ X ,    0    ]]

def build_XX(
    X, headers: Sequence[str]
):
    if X.shape[1] != len(headers):
        raise ValueError(f"X columns mismatch: X {X.shape[1]} cols vs headers {len(headers)} cols")

    n, p = X.shape
    XX = np.zeros((p + 1, p + 1), dtype = np.float32)

    XT1 = X.sum(axis = 0) / n
    XTX = X.T @ X

    XX[0, 0] = n
    XX[0, 1:] = XT1
    XX[1:, :p] = XTX

    return XX


def build_XY(X, y):
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


def assert_unique_on_keys(df, keys, name):
    dup = df.duplicated(subset = keys, keep = False)
    if dup.any():
        raise ValueError(f"{name} not unique on keys: {dup.sum()} duplicate rows")



def main():
    t0 = time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required = True, help = "YYYYMMDD, e.g. 2025-31")
    ap.add_argument("--output_path", required = True, help = "Output directory (must already exists).")
    args = ap.parse_args()

    date = args.date
    output_path = Path(args.output_path)
    if not output_path.is_dir():
        raise ValueError(f"--output_path must exist and be a directory: {output_path}")

    print(f"[START] date = {date}")

    # Inputs
    features_path = os.path.join(base_dir, features_dir, date + ".feather")
    targets_path = os.path.join(base_dir, targets_dir, date + ".feather")

    # Outputs
    headers_tmp_path = output_path / f"FeatureHeaders.{date}.tmp"
    headers_final_path = output_path / f"FeatureHeaders.{date}"

    XX_tmp_path = output_path / f"XX.{date}.csv.tmp"
    XX_final_path = output_path / f"XX.{date}.csv"

    corr_tmp_path = output_path / f"FeatureCorr.{date}.csv.tmp"
    corr_final_path = output_path / f"FeatureCorr.{date}.csv"

    # Check Existence
    exists = {
        "headers": headers_final_path.exists(),
        "xx": XX_final_path.exists(),
        "corr": corr_final_path.exists()
    }
    n_exists = sum(exists.values())

    if n_exists == 3:
        print(f"[SKIP] date = {date} already processed\n")
        return

    if n_exists != 0:
        missing = [k for k, v in exists.items() if not v]
        present = [k for k, v in exists.items() if v]
        raise RuntimeError(
            f"Partial outputs for date = {date}. {present} Present, {missing} Missing."
        )

    # Read
    features = pd.read_feather(features_path)
    targets = pd.read_feather(targets_path)

    try:
        with open(reference_headers_path, "r", encoding = "utf-8") as f:
            ref_headers = [line.rstrip("\n") for line in f]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Reference Headers Not Found: make sure FeatureHeaders.00000000 exists in cnfut directory"
        ) from e

    # Sanity Checks
    keys = ['ukey', 'ticktime', 'DataDate', 'TimeStamp']
    assert_unique_on_keys(features, keys, "Features")

    # --------------------- Main Logic ---------------------

    # Headers
    headers = build_feature_headers(features_path, prefix = "cnfut")

    if headers != ref_headers:
        raise ValueError("Features Mismatch: requires same order for headers vs ref_headers")

    write_headers(headers_tmp_path, headers)

    t1 = time.perf_counter()
    print(f"Built Headers. Elapsed: {(t1 - t0):.3f}s")

    # XX
    X = features.loc[:, headers].to_numpy(dtype = np.float32, copy = False)
    XX = build_XX(X, headers)
    np.savetxt(XX_tmp_path, XX, delimiter = ",")

    t2 = time.perf_counter()
    print(f"Built XX. Elapsed: {(t2 - t1):.3f}s")

    # XY / Corr
    # TODO: Extend
    target_cols = ["y60r05"]
    X_all = features.loc[:, headers].to_numpy(copy = False)
    p = X_all.shape[1]

    xy = pd.DataFrame({
        "ukey": np.full(p, 99999999),
        "fid": [f"F{i}" for i in range(p)]
    })

    for k, tcol in enumerate(target_cols):
        yk = targets[keys + [tcol]].dropna(subset = [tcol])
        assert_unique_on_keys(yk, keys, "Targets")

        data = features.merge(yk, on = keys, how = "inner", validate = "one_to_one")
        X = data.loc[:, headers].to_numpy(copy = False)
        y = data[tcol].to_numpy(copy = False)

        n, p2 = X.shape
        if p2 != p:
            raise ValueError(f"Header Mismatch: Expected {p} Cols, Found {p2}")

        prefix = f"y{k}."
        stats = build_XY(X, y)
        xy[prefix + "N"] = stats["N"]
        xy[prefix + "X"] = stats["X"]
        xy[prefix + "Y"] = stats["Y"]
        xy[prefix + "XX"] = stats["XX"]
        xy[prefix + "XY"] = stats["XY"]
        xy[prefix + "YY"] = stats["YY"]
        xy[prefix + "cor"] = stats["cor"]

    xy.to_csv(corr_tmp_path, index = False)

    t3 = time.perf_counter()
    print(f"Built XY / Corr. Elapsed: {(t3 - t2):.3f}s")

    # Final Rename
    os.replace(headers_tmp_path, headers_final_path)
    os.replace(XX_tmp_path, XX_final_path)
    os.replace(corr_tmp_path, corr_final_path)

    print(f"Finished {date}. Total {(t3 - t0):.3f}s\n")


if __name__ == "__main__":
    main()
