# Transformer lesson data

The active notebook and `yelp_data.py` use paths anchored to this lesson, not
machine-specific absolute paths or `legacy/`.

| Directory | Contents | Included in Git? |
| --- | --- | --- |
| `tokenizers/` | Yelp tokenizers and compressed TinyStories tokenizer | Yes |
| `models/` | Compressed TinyStories training checkpoint (archived lessons) | Yes |
| `images/` | Images used by archived vision lessons | Yes |
| `datasets/yelp_polarity/` | Downloaded raw Yelp train/test splits | No; created on first use |
| `tokenizers/yelp_bert_pruned_tokenizer_len*/train_tokens/`, `test_tokens/` | Generated tokenized Yelp splits | No; created on first use |
| `cache/huggingface/` | Downloaded BERT models/tokenizers and dataset download caches | No; created on first use |

The extracted TinyStories `.pt` and `.json` files are also ignored. Archived
TinyStories notebooks restore them from the tracked `.gz` files when needed.
Those gzip files contain a checkpoint/tokenizer, **not** the raw TinyStories dataset.
Raw datasets are downloaded from Hugging Face; no raw dataset archive is bundled.

## Fresh machine

1. Clone the repository and install [uv](https://docs.astral.sh/uv/).
2. Open a terminal in `5. Transformer` and run `launch.bat` (Windows) or
   `bash launch.sh` (Linux/macOS).
3. Run `bert_lesson.ipynb` from the beginning. Internet access is required for
   the first BERT and Yelp download. The shipped pruned Yelp tokenizer is reused;
   the tokenized train/test splits are generated once and saved here.
4. Subsequent runs reuse downloaded BERT files and the local raw/tokenized Yelp
   datasets. For a deliberately offline run, set `HF_HUB_OFFLINE=1` before launch.

For a new max-length/pruning configuration, `yelp_data.py` builds a tokenizer
and caches in a separate parameter-specific directory (and downloads the base
BERT tokenizer on first use if needed).

## Checks

Run `uv run --extra cuda python -m unittest discover -p "test_*.py"` (use
`--extra xpu` on Intel GPU systems). `test_data_setup.py` exercises a fresh copy
with only tracked tokenizer inputs and no `legacy/`, first-use tokenization,
offline cache reuse, raw-dataset persistence, and gzip extraction. It uses tiny
fixture reviews, not a full network download or training run.

The end-to-end acceptance check is a fresh clone on another machine, running
the notebook top-to-bottom, then rerunning with `HF_HUB_OFFLINE=1`. Existing
`.venv`, user-level Hugging Face caches, and generated data must not be copied
for that check.
