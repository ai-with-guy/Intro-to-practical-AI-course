# PyTorch Training Memory

Where GPU memory goes during training: weights, gradients, optimizer state, and
activations — and the four practical knobs (batch size + accumulation, mixed
precision, checkpointing, optimizer choice) when you run out.

## Launch

- Windows: run `launch.bat`.
- Linux/macOS: run `bash launch.sh`.

The launchers use `uv`, create a lesson-local environment, and open
`training_memory.ipynb`. They follow lessons 4–6's CUDA/XPU selection
(Intel-only GPUs select XPU on Windows/Linux). CPU execution works with the
default build too; the notebook chooses CUDA, XPU, MPS, or CPU depending on
availability. Override the build with `TORCH_EXTRA=cuda` or
`TORCH_EXTRA=xpu` if needed. Python 3.11–3.13 is supported.

No downloads are needed: the notebook trains a tiny synthetic MLP, so the
first run only installs PyTorch and Jupyter. Internet is needed once for that;
subsequent runs are offline. `memory_helpers.py` holds the reusable estimation
helpers; the notebook plots and measures them.

## Checks

From this directory:

```bash
uv run --extra cuda python -m unittest -v test_memory_helpers
```

Use `--extra xpu` if appropriate. Tests run on CPU with a tiny MLP — no
checkpoint or image downloads. They check parameter/gradient/optimizer math,
linear batch scaling of activations, and that every notebook cell compiles.
