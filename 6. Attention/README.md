# 6. Attention

An extension of **5. Transformer**. The first worked example is Java → C#
translation with `Salesforce/codet5-base-codexglue-translate-java-cs`:
generate code, inspect cross-attention, then compare encoder and causal decoder
self-attention. A second example translates German → English with
`Helsinki-NLP/opus-mt-de-en`, using the same viewer and a heatmap labeled with
decoder queries and next-token predictions.

## Launch

- Windows: run `launch.bat`.
- Linux/macOS: run `bash launch.sh`.

The launchers use `uv`, create a lesson-local environment, and open
`attention.ipynb`. They follow lesson 5's CUDA/XPU selection (Intel-only GPUs
select XPU on Windows/Linux). CPU execution works with the default build too;
the notebook chooses CUDA, XPU, MPS, or CPU depending on availability.
Override the build with `TORCH_EXTRA=cuda` or `TORCH_EXTRA=xpu` if needed.
Python 3.11–3.13 is supported.

The first run downloads PyTorch and the pretrained checkpoint (roughly 900 MB
for the model weights), plus the tokenizer from `Salesforce/codet5-base`.
The translation checkpoint does **not** contain tokenizer files. Internet is
needed for the first download; subsequent runs use the Hugging Face cache.
No training or dataset download is required. Generated code is not executed and
may be wrong—review it as you would any model-generated code.

The German → English section downloads a separate Marian checkpoint (roughly
300 MB). SentencePiece and Sacremoses are included for its tokenizer.

The notebook also normalizes the base tokenizer's legacy sentinel-token metadata
for Transformers 5, without changing the vocabulary IDs.

## Shared BertViz fork

The notebook imports `../5. Transformer/bertviz_local` and its bundled
`bertviz_assets`, rather than installing upstream BertViz or copying the fork.
Keep both lesson directories together. The same iframe-based rendering and
local JavaScript assets used in lesson 5 are used here; no frontend CDN is needed.
The upstream and bundled-library licenses remain in lesson 5.

Both launchers set `PYTHONPATH` before starting Jupyter, so the notebook uses a
plain import. If you see `No module named 'bertviz_local'`, stop the old Jupyter
server and launch again with `launch.bat` / `launch.sh`; restarting only its kernel
does not update the server's inherited environment.

### Generation replay

`generation_view(german_example, layer=3, heads=[2])` shows **read → predict →
append** one token at a time, with Next/Back, Play/Pause, Reset, and a step slider.
Decoder attention is cropped to the current prefix; cross-attention keeps the
full source. The latest query is highlighted. This uses recorded generation and
attention, without rerunning the model. The Python function defaults are unchanged.

## Checks

From this directory:

```bash
uv run --extra cuda python -m unittest -v test_attention_notebook
```

Use `--extra xpu` if appropriate. Tests use a tiny randomly initialized T5 and
a synthetic tokenizer, so they do not download the lesson's model. They check
generation/replay alignment, cross-attention shapes, masks, and BertViz output.
