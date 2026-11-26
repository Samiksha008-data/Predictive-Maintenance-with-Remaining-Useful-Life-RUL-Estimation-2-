from pathlib import Path
import pandas as pd

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
ROOT_DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def _resolve_data_dir() -> Path:
    for candidate in (PACKAGE_DATA_DIR, ROOT_DATA_DIR, Path(__file__).resolve().parents[1] / "data"):
        if (candidate / "train_FD001.txt").exists():
            return candidate
    return ROOT_DATA_DIR


def load_data(dataset: str = "FD001"):
    """Load raw training, test, and RUL files for a C-MAPSS subset.

    Returns:
        train_df (pd.DataFrame): training trajectories
        test_df (pd.DataFrame): test trajectories
        rul_series (pd.Series): true RUL after last test cycle for each test unit
    """
    data_dir = _resolve_data_dir()
    train_file = data_dir / f"train_{dataset}.txt"
    test_file = data_dir / f"test_{dataset}.txt"
    rul_file = data_dir / f"RUL_{dataset}.txt"

    if not (train_file.exists() and test_file.exists() and rul_file.exists()):
        raise FileNotFoundError(
            f"Required C-MAPSS files not found in {data_dir}. "
            f"Please place 'train_{dataset}.txt', 'test_{dataset}.txt', and 'RUL_{dataset}.txt' there."
        )

    col_names = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=col_names)
    rul_series = pd.read_csv(rul_file, sep=r"\s+", header=None).squeeze("columns")
    if isinstance(rul_series, pd.DataFrame):
        rul_series = rul_series.iloc[:, 0]
    return train_df, test_df, rul_series
