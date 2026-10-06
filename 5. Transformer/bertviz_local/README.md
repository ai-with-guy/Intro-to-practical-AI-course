# Attention flow view

Derived from the BertViz 1.4.1 head view (Apache-2.0; see `LICENSE`), with
layer/head controls and token hover behavior preserved. It uses
bundled jQuery/D3 rather than `require.js`, renders in an isolated iframe, and
accepts attention weights from custom models, including single-head attention.

The frontend is `attention_flow.js`; the Python API stays `bertviz_local.head_view`
for notebook compatibility. Cross-attention is selected initially whenever supplied.
Animated dots travel **right to left**, from keys/values being read to queries
receiving information (encoder to decoder for cross-attention). These show value
flow, not the opposite direction of a query looking up a key. Brighter dots/lines
indicate stronger attention. Hover filtering and head selection apply to both.

Use **Animate flow** to stop/start motion. Animation starts disabled for users
with reduced-motion preferences. At most 300 of the strongest active connections
get dots to keep long sequences responsive; all attention lines are still drawn.

## Generation replay

```python
from bertviz_local import generation_view
generation_view(example, layer=0, heads=[0])
```

`example` is the dictionary produced by lesson 6's `inspect_translation`, with
encoder/decoder tokens, `predicted_tokens`, and decoder/cross attention. The viewer
replays **read prefix → predict next token → append token**, with manual steps,
playback, reset, and a slider. Decoder self-attention is cropped to the available
prefix; cross-attention retains every encoder token. The final query is highlighted.
This is a replay of recorded greedy generation, not live inference or token
probabilities. Predictions must align with decoder inputs shifted by one position;
the last appended token was not read in the recording and has no query row.

## Custom model attention

Pass attention weights after softmax, with rows representing query tokens and
columns representing key tokens. No Hugging Face model or output object is needed:

```python
from bertviz_local import head_view

# attention_scores: [tokens, tokens], as in the notebook's final visualization
head_view(attention_scores, words)

# Multiple simple-attention layers: a list of [tokens, tokens] matrices
head_view([layer_0_scores, layer_1_scores], words)
```

Each layer can be a PyTorch tensor, NumPy array, or nested Python list:

- `[queries, keys]`: one head.
- `[heads, queries, keys]`: unbatched multi-head attention, or single-head
  attention with a singleton batch axis `[1, queries, keys]`.
- `[1, heads, queries, keys]`: the original BertViz/Hugging Face format.

A single tensor/array represents one layer; a list/tuple of matrices represents
multiple layers. A nested numeric matrix is also accepted directly. Select one
sample from larger batches before calling `head_view`; for simple attention use
`scores[example_index]`. Three-dimensional inputs are interpreted as heads, not
as multiple batch examples. Tensors are detached and moved to CPU automatically.
All layers must have matching shapes, and token counts must match the matrix.
Single-head inputs appear as **head 0** in the existing UI.

The same formats work for `encoder_attention`, `decoder_attention`, and
`cross_attention` (whose query and key lengths can differ). Existing layer/head
controls, sentence-pair views, and `html_action='return'` remain available.

The bundled libraries in `../bertviz_assets/` are jQuery 2.0.0 (MIT) and
D3 5.7.0 (BSD-3-Clause); their licenses are alongside the files.
