"""First-run and offline rerun checks, using tiny datasets instead of full training."""

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

import data_paths
import yelp_data
from transformers import AutoTokenizer


class DataSetupTests(unittest.TestCase):
    def test_missing_tokenizer_is_built_and_reused(self):
        original = AutoTokenizer.from_pretrained(
            data_paths.TOKENIZERS_DIR / "yelp_bert_pruned_tokenizer_len512_rare5",
            local_files_only=True,
        )
        dataset = DatasetDict({
            split: Dataset.from_dict({"text": ["great food", "bad food"], "label": [1, 0]})
            for split in ("train", "test")
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(yelp_data, "TOKENIZERS_DIR", root), \
                    patch("yelp_data.load_yelp_dataset", return_value=dataset) as download, \
                    patch("yelp_data.AutoTokenizer.from_pretrained", return_value=original) as base:
                tokenizer, _, test = yelp_data.load_yelp_data(
                    max_len=16, rare_token_limit=0, batch_size=2)
                self.assertEqual(download.call_count, 1)
                base.assert_called_once_with(
                    "bert-base-uncased", use_fast=True, cache_dir=str(data_paths.HF_MODELS_CACHE))
                ids, _ = next(iter(test))
                self.assertLessEqual(ids.shape[1], 16)
                self.assertLess(int(ids.max()), len(tokenizer))
            with patch("yelp_data.load_yelp_dataset", side_effect=AssertionError("No download")):
                again, _, _ = yelp_data.load_yelp_data(
                    max_len=16, rare_token_limit=0,
                    directory=root / "yelp_bert_pruned_tokenizer_len16_rare0")
                self.assertEqual(tokenizer.get_vocab(), again.get_vocab())

    def test_raw_dataset_download_then_offline_reuse(self):
        dataset = DatasetDict({
            split: Dataset.from_dict({"text": ["great food", "bad food"], "label": [1, 0]})
            for split in ("train", "test")
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(data_paths, "DATASETS_DIR", root / "datasets"), \
                    patch.object(data_paths, "HF_DATASETS_CACHE", root / "cache"), \
                    patch("datasets.load_dataset", return_value=dataset) as download:
                first = data_paths.load_yelp_dataset()
                download.assert_called_once_with("ajay232/yelp_polarity", cache_dir=str(root / "cache"))
                download.side_effect = AssertionError("Rerun must not download")
                second = data_paths.load_yelp_dataset()
                self.assertEqual(first["train"][:], second["train"][:])

    def test_fresh_copy_without_legacy_or_existing_caches(self):
        lesson = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data_paths.py", "yelp_data.py"):
                shutil.copy2(lesson / name, root / name)
            shipped = "yelp_bert_pruned_tokenizer_len512_rare5"
            shutil.copytree(lesson / "data" / "tokenizers" / shipped,
                            root / "data" / "tokenizers" / shipped,
                            ignore=shutil.ignore_patterns("train_tokens", "test_tokens"))
            script = '''
from pathlib import Path
from unittest.mock import patch
from datasets import Dataset, DatasetDict
import yelp_data
dataset = DatasetDict({split: Dataset.from_dict({
    "text": ["great food", "bad food"], "label": [1, 0]
}) for split in ("train", "test")})
with patch("yelp_data.load_yelp_dataset", return_value=dataset) as download:
    tokenizer, train, test = yelp_data.load_yelp_data(batch_size=2)
    assert download.call_count == 1
    ids, labels = next(iter(test))
    assert ids.shape[0] == 2 and labels.tolist() == [1, 0]
    download.side_effect = AssertionError("Offline rerun must use token caches")
    again, _, test_again = yelp_data.load_yelp_data(batch_size=2)
    assert tokenizer.get_vocab() == again.get_vocab()
    assert (ids == next(iter(test_again))[0]).all()
assert not Path("legacy").exists()
assert (Path("data/tokenizers") / "yelp_bert_pruned_tokenizer_len512_rare5/train_tokens/dataset_info.json").is_file()
'''
            result = subprocess.run([sys.executable, "-c", script], cwd=root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gzip_first_run_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            with gzip.open(str(path) + ".gz", "wb") as target:
                target.write(b'{"test": true}')
            self.assertEqual(data_paths.extract_if_missing(path).read_bytes(), b'{"test": true}')
            with patch("data_paths.gzip.open", side_effect=AssertionError("Already extracted")):
                self.assertEqual(data_paths.extract_if_missing(path), path)

    def test_notebook_code_is_valid_and_does_not_read_legacy_data(self):
        lesson = Path(__file__).resolve().parent
        for notebook in [lesson / "bert_lesson.ipynb", * (lesson / "legacy").glob("*.ipynb")]:
            for i, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    # Archived notebooks can contain IPython magics.
                    if not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                        compile(source, f"{notebook.name}:cell{i}", "exec")
                    if notebook.name == "bert_lesson.ipynb":
                        self.assertNotIn("legacy", source)


if __name__ == "__main__":
    unittest.main()
