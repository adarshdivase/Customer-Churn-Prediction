"""Pre-train and cache the XGBoost pipeline for faster demo startup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from churnai.config import DEFAULT_DATA, PIPELINE_PATH
from churnai.model import train_pipeline


def main():
    path = Path(DEFAULT_DATA)
    if not path.exists():
        print(f"Dataset not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    result = train_pipeline(df, use_cache=False, mode="accurate", calibrate=True)
    if result:
        print(f"Saved pipeline to {PIPELINE_PATH}")
    else:
        print("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
