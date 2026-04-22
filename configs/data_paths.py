from dataclasses import dataclass
from pathlib import Path 


@dataclass
class DataPaths:
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("./data/processed_data")
    mpds_data_file: Path = raw_data_dir / "mpds_data.pkl.gz"
    perovskite_cifs_dir: Path = raw_data_dir / "perovskite_cifs"
    processed_cifs_file: Path = processed_data_dir / "processed_cifs.pkl.gz"
    train_data_file: Path = processed_data_dir / "train_data.pkl.gz"
    test_data_file: Path = processed_data_dir / "test_data.pkl.gz"  
    val_data_file: Path = processed_data_dir / "val_data.pkl.gz"
    logs_dir: Path = Path("logs")
    tokens_folder = Path("data/tokens")
    train_tokens_file = tokens_folder / "train_tokens.pkl.gz"
    val_tokens_file = tokens_folder / "val_tokens.pkl.gz"
    test_tokens_file = tokens_folder / "test_tokens.pkl.gz"