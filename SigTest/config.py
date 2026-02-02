from pathlib import Path

class ValidationConfig:

    # === Paths ===
    feature_path: Path
    target_path: Path
    meta_path: Path
    calendar_path: Path 
    output_path: Path

    # === Backtest parameters ===
    asof: int
    lookback: int
    lookforward: int
    target_cols: list[str]

    # === Method selection ===
    methods: list[str]

    # === Hyperparameters ===
    hyperparams: dict
