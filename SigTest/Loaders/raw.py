from pathlib import Path
import numpy as np
import pandas as pd

from SigTest.config import (
    JOIN_KEYS,
    DEFAULT_FEATURE_PATH,
    DEFAULT_TARGET_PATH,
    DEFAULT_REF_HEADERS_PATH,
)


def read_headers(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class FeatherLoader:
    def __init__(
        self,
        feature_path: Path = DEFAULT_FEATURE_PATH,
        target_path: Path = DEFAULT_TARGET_PATH,
        ref_headers_path: Path = DEFAULT_REF_HEADERS_PATH,
    ):
        self.feature_path = Path(feature_path)
        self.target_path = Path(target_path)
        self.ref_headers = read_headers(ref_headers_path)

  
    def load_day(
        self,
        date: int,
        features: str | list[str] = "full",
        targets: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feat_file = self.feature_path / f"{date}.feather"
        targ_file = self.target_path / f"{date}.feather"

        if not feat_file.exists():
            raise FileNotFoundError(f"Feature file missing: {feat_file}")
        if not targ_file.exists():
            raise FileNotFoundError(f"Target file missing: {targ_file}")

        if features == "full":
            feat_cols = self.ref_headers
            feather_cols = pd.read_feather(feat_file, columns=None).columns.tolist()
            feather_feat_cols = [c for c in feather_cols if c not in JOIN_KEYS]
            if feather_feat_cols != self.ref_headers:
                raise ValueError(
                    f"Feature headers mismatch at {date}: "
                    f"expected {len(self.ref_headers)} columns in ref_headers order"
                )
        else:
            feat_cols = features

        features_df = pd.read_feather(feat_file, columns=JOIN_KEYS + feat_cols)

        if targets is not None:
            targets_df = pd.read_feather(targ_file, columns=JOIN_KEYS + targets)
        else:
            targets_df = pd.read_feather(targ_file)

        return features_df, targets_df

  
    def load_window(
        self,
        dates: list[int],
        features: str | list[str] = "full",
        targets: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not dates:
            raise ValueError("dates list is empty")

        feat_list = []
        targ_list = []
        for d in dates:
            f, t = self.load_day(d, features=features, targets=targets)
            feat_list.append(f)
            targ_list.append(t)

        features_df = pd.concat(feat_list, ignore_index=True)
        targets_df = pd.concat(targ_list, ignore_index=True)

        return features_df, targets_df

  
    def merge_target(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        target_col: str,
        feature_cols: str | list[str] = "full",
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature_cols == "full":
            feature_cols = self.ref_headers
            df_feat_cols = [c for c in features.columns if c not in JOIN_KEYS]
            if df_feat_cols != self.ref_headers:
                raise ValueError(
                    f"Feature headers mismatch: "
                    f"expected {len(self.ref_headers)} columns in ref_headers order"
                )

        targ = targets[JOIN_KEYS + [target_col]].dropna(subset=[target_col])

        feat_idx = pd.util.hash_pandas_object(features[JOIN_KEYS], index=False).values
        targ_idx = pd.util.hash_pandas_object(targ[JOIN_KEYS], index=False).values

        if len(feat_idx) != len(np.unique(feat_idx)):
            raise ValueError("Duplicate keys in features")
        if len(targ_idx) != len(np.unique(targ_idx)):
            raise ValueError("Duplicate keys in targets")

        common, feat_pos, targ_pos = np.intersect1d(feat_idx, targ_idx, return_indices=True)

        X = features.iloc[feat_pos][feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = targ.iloc[targ_pos][target_col].to_numpy(dtype=np.float64, copy=False)

        return X, y
