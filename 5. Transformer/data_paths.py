"""Lesson-local data paths, independent of the notebook's working directory."""

import gzip
from pathlib import Path
import shutil


DATA_DIR = Path(__file__).resolve().parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
MODELS_DIR = DATA_DIR / "models"
HF_DATASETS_CACHE = DATA_DIR / "cache" / "huggingface" / "datasets"
HF_MODELS_CACHE = DATA_DIR / "cache" / "huggingface" / "hub"


def extract_if_missing(path):
    """Restore a generated asset from its tracked gzip copy on first use."""
    path = Path(path)
    if not path.is_file():
        compressed = path.with_suffix(path.suffix + ".gz")
        with gzip.open(compressed, "rb") as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return path


def load_yelp_dataset():
    """Download once; reuse the local raw dataset without contacting the Hub."""
    from datasets import load_dataset, load_from_disk

    directory = DATASETS_DIR / "yelp_polarity"
    if (directory / "dataset_dict.json").is_file():
        return load_from_disk(str(directory))
    dataset = load_dataset("ajay232/yelp_polarity", cache_dir=str(HF_DATASETS_CACHE))
    dataset.save_to_disk(str(directory))
    return dataset
