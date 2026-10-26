#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
launch.bat
```

The launchers select `cuda` when an NVIDIA GPU is present. On systems with Intel
graphics and no NVIDIA GPU, they select the `xpu` dependency extra from
`pyproject.toml`. The XPU extra uses PyTorch's official XPU wheel index and
requires a supported Intel GPU and current graphics drivers.

### Training timelines

In `pytorch_tensors_and_numpy.ipynb`, classifier displays start at the final epoch.
Drag **Epoch** backward/forward to inspect every saved step, including the initial
weights at epoch 0. Snapshots stay on the model's device; only the selected
prediction is copied to the CPU for plotting.

The final playground stays live-only, without snapshots or an epoch slider.
Classifier snapshots contain model weights, not optimizer checkpoints or
pre-rendered images.

Run snapshot tests with `python -m unittest test_training_history -v`.
