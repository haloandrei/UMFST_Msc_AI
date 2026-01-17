#!/usr/bin/env python3
import os
from pathlib import Path

import kagglehub


def resolve_training_path(data_path: str) -> str:
    candidates = [
        Path(data_path) / "fruits-360" / "Training",
        Path(data_path) / "Training",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    raise FileNotFoundError(f"Could not find: {data_path}")


def main() -> None:
    data_path = kagglehub.dataset_download("moltean/fruits")
    training_path = resolve_training_path(data_path)

if __name__ == "__main__":
    main()
